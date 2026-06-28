import os
import pandas as pd
import config


def flatten_multirow_columns(df):
    """
    Spljošćuje dvoredi (multi-row) Excel header u jednoznačne string
    nazive kolona, npr.:
      ('Progression Status', 'PLR2')      -> 'Progression Status_PLR2'
      ('VF', 0)                            -> 'VF_0'
      ('Subject Number', 'Unnamed: ...')   -> 'Subject Number'

    Ovo je potrebno jer pandas/openpyxl NE podržava snimanje MultiIndex
    kolona u Excel kada je index=False (NotImplementedError), pa se mora
    flatten-ovati PRE čuvanja na disk. Pošto se flatten radi na isti,
    deterministički način svuda u pipeline-u (ovde i u
    create_gru_sequences.py), informacija iz originalnog dvorednog
    headera se ne gubi — samo se predstavlja kao ravni string.
    """
    new_cols = []
    for top, bottom in df.columns:
        top = str(top).strip()
        bottom = str(bottom).strip()
        if top == "VF":
            new_cols.append(f"VF_{bottom}")
        elif bottom.startswith("Unnamed") or bottom in ("", "nan"):
            new_cols.append(top)
        else:
            new_cols.append(f"{top}_{bottom}")
    df.columns = new_cols
    return df


def main():
    grape_excel_path = os.path.join(config.DATA_DIR, "VF and clinical information.xlsx")
    unet_features_path = os.path.join(config.OUTPUT_DIR, "grape_extracted_features.xlsx")

    output_baseline_path = os.path.join(config.OUTPUT_DIR, "grape_baseline_merged.xlsx")
    output_followup_path = os.path.join(config.OUTPUT_DIR, "grape_followup_merged.xlsx")

    if not os.path.exists(grape_excel_path):
        print(f"[GREŠKA] Nije pronađen GRAPE Excel fajl na putanji: {grape_excel_path}")
        return
    if not os.path.exists(unet_features_path):
        print(f"[GREŠKA] Nije pronađen EXCEL sa UNet osobinama na putanji: {unet_features_path}")
        return

    print("-> Učitavanje podataka...")
    unet_df = pd.read_excel(unet_features_path)

    # ISPRAVKA: čitamo sa header=[0, 1] (dvoredi/multi-row header), ne
    # default header=0. Originalni Excel ima spojene ćelije (npr.
    # "Progression Status" je nadzaglavlje za PLR2/PLR3/MD, "VF" je
    # nadzaglavlje za 61 kolonu vrednosti). Sa header=0, pandas te
    # pod-kolone čita kao "Unnamed: N", što je fragilno i zavisi od
    # tačnog broja kolona ispred — male promene uzvodno tiho pomeraju
    # koja "Unnamed" kolona odgovara PLR2 vs PLR3 vs MD (ovo je bio bug
    # u prethodnoj verziji pipeline-a).
    df_baseline = pd.read_excel(grape_excel_path, sheet_name=0, header=[0, 1])
    df_followup = pd.read_excel(grape_excel_path, sheet_name=1, header=[0, 1])

    print(f"-> Broj zapisa pre spajanja - Baseline: {len(df_baseline)}, Follow-up: {len(df_followup)}")

    df_baseline = flatten_multirow_columns(df_baseline)
    df_followup = flatten_multirow_columns(df_followup)

    print("-> Spajanje UNet parametara sa Sheet 1 (Baseline)...")
    baseline_merged = pd.merge(df_baseline, unet_df, on="Corresponding CFP", how="left")

    print("-> Spajanje UNet parametara sa Sheet 2 (Follow-up)...")
    followup_merged = pd.merge(df_followup, unet_df, on="Corresponding CFP", how="left")

    unet_cols = ["vCDR", "hCDR", "aCDR", "Rim_Area_Pixels"]

    # NAPOMENA: GRAPE follow-up sheet koristi '/' u koloni "Corresponding
    # CFP" kada ta poseta NIJE imala urađen fundus foto (npr. poseta je
    # bila samo IOP/VF kontrola, bez slikanja). To je OČEKIVAN, klinički
    # legitiman slučaj — ne greška u merge-u. Razdvajamo ga od "stvarnog"
    # promašaja (CFP naziv postoji, ali se ne poklapa sa UNet fajlom),
    # jer ta dva slučaja zahtevaju različitu reakciju: prvi je normalan i
    # ne treba upozorenje, drugi ukazuje na bug koji vredi istražiti.
    NO_CFP_MARKER = "/"

    for name, df_merged, df_original in [
        ("Baseline", baseline_merged, df_baseline),
        ("Follow-up", followup_merged, df_followup),
    ]:
        no_cfp_mask = df_original["Corresponding CFP"].astype(str).str.strip() == NO_CFP_MARKER
        missing_unet = df_merged["vCDR"].isna()

        n_no_cfp = no_cfp_mask.sum()
        n_real_mismatch = (missing_unet & ~no_cfp_mask.values).sum()

        print(f"-> {name}: {n_no_cfp} poseta bez CFP slike (marker '/', očekivano), {n_real_mismatch} pravih promašaja u merge-u (neočekivano).")
        if n_real_mismatch > 0:
            print(f"   [UPOZORENJE] {n_real_mismatch} redova u {name} ima naziv CFP slike koji NE postoji u UNet feature fajlu — vredi proveriti zašto (npr. slika nije obrađena ekstrakcijom).")

        # has_cfp = 1.0 ako poseta ima validnu CFP sliku I UNet je uspešno
        # izvukao parametre iz nje; 0.0 inače (bilo da nema slike, bilo
        # da je merge promašio). Ovo je feature koji ulazi u GRU da
        # model nauči da razlikuje "stvarno izmereno" od "nedostaje".
        df_merged["has_cfp"] = (~missing_unet).astype(float)

    # Popunjavamo vCDR/hCDR/aCDR/Rim_Area_Pixels sa 0.0 (NE medijanom)
    # kada CFP nije dostupan. Razlog: medijana bi predstavila izmišljenu
    # ali "plauzibilnu" vrednost kao da je stvarno izmerena, što je
    # posebno problematično kada se to dešava za ~43% follow-up poseta
    # (vidi dijagnostiku) — model bi učio na masovno izmišljenim
    # vrednostima. Sa 0.0 + has_cfp=0.0 kao eksplicitan signal,
    # mreža kroz trening uči da ignoriše ove kolone kada flag kaže da
    # podatak ne postoji, umesto da uči lažnu korelaciju iz konstante.
    baseline_merged[unet_cols] = baseline_merged[unet_cols].fillna(0.0)
    followup_merged[unet_cols] = followup_merged[unet_cols].fillna(0.0)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    baseline_merged.to_excel(output_baseline_path, index=False)
    followup_merged.to_excel(output_followup_path, index=False)

    print("\n================ SPAJANJE PODATAKA USPEŠNO ================")
    print(f"1. Baseline tabela sačuvana na:  {output_baseline_path}")
    print(f"2. Follow-up tabela sačuvana na: {output_followup_path}")
    print(f"Dodata obeležja: {unet_cols + ['has_cfp']}")
    print("==========================================================")


if __name__ == "__main__":
    main()