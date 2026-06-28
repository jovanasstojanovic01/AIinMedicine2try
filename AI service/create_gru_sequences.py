import os
import re
import numpy as np
import pandas as pd
import config

# Slepe tačke i nepouzdana merenja su markirana sa -1 u GRAPE VF kolonama.
VF_BLIND_SPOT_VALUE = -1

# Pattern za flatten-ovane VF kolone koje pravi merge_grape_data.py
# (flatten_multirow_columns): "VF_0", "VF_1", ..., "VF_60".
VF_COLUMN_PATTERN = re.compile(r"^VF_(\d+)$")

# JEDAN zajednički izvor istine za listu feature-a koji ulaze u GRU.
# train.py uvozi OVU konstantu (umesto da je duplira lokalno), da se ne
# ponovi bug iz ranije verzije pipeline-a gde su dve skripte mapirale
# isti Excel na različita imena kolona zbog nezavisnih, nesinhronizovanih
# definicija.
# JEDAN zajednički izvor istine za listu feature-a koji ulaze u GRU.
# train.py uvozi OVU konstantu (umesto da je duplira lokalno), da se ne
# ponovi bug iz ranije verzije pipeline-a gde su dve skripte mapirale
# isti Excel na različita imena kolona zbog nezavisnih, nesinhronizovanih
# definicija.
#
# 'Interval_Years' = vreme (u godinama) od PRETHODNE posete. Baseline
# (prva poseta) ima 0.0 po definiciji — referentna tačka u vremenu. Ovaj
# feature je bitan jer VF promene zavise od toga koliko je vremena
# prošlo od prethodnog merenja — bez njega, model ne razlikuje "sledeća
# poseta za 2 meseca" od "sledeća poseta za 2 godine".
FEATURES_LIST = ["IOP", "vCDR", "hCDR", "aCDR", "Rim_Area_Pixels", "has_cfp", "Interval_Years", "VF_mean"]


def get_vf_columns(df):
    """
    Vraća listu flatten-ovanih VF kolona ('VF_0'...'VF_60'), sortiranih
    po numeričkom indeksu (ne leksikografski, da 'VF_10' ne dođe pre
    'VF_2').
    """
    cols = [c for c in df.columns if VF_COLUMN_PATTERN.match(str(c))]
    cols.sort(key=lambda c: int(VF_COLUMN_PATTERN.match(c).group(1)))
    return cols


def compute_vf_mean(df, vf_cols):
    """
    Računa prosečnu svetlosnu osetljivost vidnog polja (VF) po redu,
    isključujući slepe tačke (-1) i bilo koji NaN.

    Ovo je proxy za MD (mean deviation): pravi MD bi bio prosek TD
    vrednosti (izmereno - normativno_po_starosti), ali GRAPE Excel ne
    sadrži starosno-normirane vrednosti niti gotovu MD kolonu po
    follow-up poseti — jedina "MD" kolona u fajlu je binarni progression
    flag na baseline-u (Progression Status_MD), ne kontinuelna vrednost
    u dB, i ne postoji nikakva MD vrednost na follow-up sheetu. Prosečna
    sirova senzitivnost preko validnih lokacija je monotono povezana sa
    funkcionalnim propadanjem vida i ne zahteva pretpostavljenu
    normativnu tabelu koju ne imamo.
    """
    vf_block = df[vf_cols].apply(pd.to_numeric, errors="coerce")
    masked = vf_block.where(vf_block != VF_BLIND_SPOT_VALUE)
    return masked.mean(axis=1, skipna=True)


def main():
    baseline_path = os.path.join(config.OUTPUT_DIR, "grape_baseline_merged.xlsx")
    followup_path = os.path.join(config.OUTPUT_DIR, "grape_followup_merged.xlsx")

    output_x_path = os.path.join(config.OUTPUT_DIR, "X_gru.npy")
    output_y_path = os.path.join(config.OUTPUT_DIR, "y_gru.npy")
    output_mask_path = os.path.join(config.OUTPUT_DIR, "mask_gru.npy")
    output_lengths_path = os.path.join(config.OUTPUT_DIR, "lengths_gru.npy")

    if not os.path.exists(baseline_path) or not os.path.exists(followup_path):
        print("[GREŠKA] Prvo moraš pokrenuti prethodni korak (merge_grape_data.py)!")
        return

    print("-> Učitavanje unifikovanih (flatten-ovanih) tabela...")
    df_b = pd.read_excel(baseline_path)
    df_f = pd.read_excel(followup_path)

    vf_cols_b = get_vf_columns(df_b)
    vf_cols_f = get_vf_columns(df_f)
    if len(vf_cols_b) == 0 or len(vf_cols_f) == 0:
        print("[GREŠKA] Nisu pronađene VF_* kolone — provери da li je merge_grape_data.py ažuriran (flatten_multirow_columns).")
        return

    print(f"-> Računanje VF_mean proxy-MD vrednosti ({len(vf_cols_b)} lokacija baseline, {len(vf_cols_f)} follow-up)...")
    df_b["VF_mean"] = compute_vf_mean(df_b, vf_cols_b)
    df_f["VF_mean"] = compute_vf_mean(df_f, vf_cols_f)

    # Baseline poseta je uvek vizit broj 0 u hronologiji pacijenta, i
    # nema "prethodnu" posetu — interval od prethodne posete je 0 po
    # definiciji (referentna tačka u vremenu).
    df_b["Visit Number"] = 0
    df_b["Interval_Years"] = 0.0

    # Follow-up sheet ima kolonu "Interval Years" (sa razmakom, kako je
    # u originalnom Excelu) — preimenujemo u "Interval_Years" da se
    # poklopi sa FEATURES_LIST i sa baseline kolonom.
    if "Interval Years" in df_f.columns:
        df_f = df_f.rename(columns={"Interval Years": "Interval_Years"})

    features_list = FEATURES_LIST
    id_cols = ["Subject Number", "Laterality", "Visit Number"]

    missing_b = [c for c in features_list if c not in df_b.columns]
    missing_f = [c for c in features_list if c not in df_f.columns]
    if missing_b or missing_f:
        print(f"[GREŠKA] Nedostaju feature kolone. Baseline: {missing_b}, Follow-up: {missing_f}")
        return

    df_b_sub = df_b[id_cols + features_list].copy()
    df_f_sub = df_f[id_cols + features_list].copy()

    df_all = pd.concat([df_b_sub, df_f_sub], axis=0, ignore_index=True)
    df_all = df_all.dropna(subset=["Subject Number", "Laterality", "Visit Number"])
    df_all = df_all.sort_values(by=["Subject Number", "Laterality", "Visit Number"]).reset_index(drop=True)

    print("-> Kreiranje per-visit sekvenci (next-step VF_mean predikcija)...")

    num_features = len(features_list)
    vf_mean_idx = features_list.index("VF_mean")

    per_eye_data = []
    max_input_steps = 0

    grouped = df_all.groupby(["Subject Number", "Laterality"])
    for (_subj, _lat), group in grouped:
        group = group.sort_values(by="Visit Number")
        feats = group[features_list].to_numpy(dtype=np.float32)

        n_visits = feats.shape[0]
        if n_visits < 2:
            # Nema "sledeće" posete da bude target — oko se ne može
            # koristiti za next-step predikciju.
            continue

        # Ulaz: posete 0..n-2 (sve osim zadnje)
        # Target: VF_mean posete 1..n-1 (svaka "sledeća" poseta, korak po korak)
        x_eye = feats[:-1, :]
        y_eye = feats[1:, vf_mean_idx]

        per_eye_data.append((x_eye, y_eye))
        max_input_steps = max(max_input_steps, x_eye.shape[0])

    if not per_eye_data:
        print("[GREŠKA] Nijedno oko nema bar 2 poseta — next-step predikcija nije moguća.")
        return

    X_sequences, y_targets, target_mask, lengths_list = [], [], [], []

    for x_eye, y_eye in per_eye_data:
        actual_steps = x_eye.shape[0]

        # RIGHT-PADDING (stvarni podaci prvi, nule na kraju). Ispravka
        # kritičnog bug-a iz prethodne verzije: padding je bio na
        # POČETKU niza (padded[-actual_steps:] = ...), ali
        # pack_padded_sequence pretpostavlja right-padding i koristi
        # 'lengths' da odseče od početka niza — sa left-padding-om je
        # efektivno odsecao stvarne podatke i ostavljao nule, tako da se
        # model trenirao na paddingu umesto na stvarnoj istoriji za sva
        # oka kraća od max_input_steps.
        padded_x = np.zeros((max_input_steps, num_features), dtype=np.float32)
        padded_y = np.zeros((max_input_steps,), dtype=np.float32)
        padded_mask = np.zeros((max_input_steps,), dtype=np.float32)

        padded_x[:actual_steps] = x_eye
        padded_y[:actual_steps] = y_eye
        padded_mask[:actual_steps] = 1.0

        X_sequences.append(padded_x)
        y_targets.append(padded_y)
        target_mask.append(padded_mask)
        lengths_list.append(actual_steps)

    X = np.array(X_sequences, dtype=np.float32)       # (N_ociju, max_steps, n_features)
    y = np.array(y_targets, dtype=np.float32)          # (N_ociju, max_steps) - VF_mean SLEDEĆE posete
    mask = np.array(target_mask, dtype=np.float32)     # (N_ociju, max_steps) - 1.0 gde y važi
    lengths = np.array(lengths_list, dtype=np.int32)   # (N_ociju,) - broj validnih ulaznih koraka

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.save(output_x_path, X)
    np.save(output_y_path, y)
    np.save(output_mask_path, mask)
    np.save(output_lengths_path, lengths)

    print("\n================ SEKVENCIRANJE ZAVRŠENO ================")
    print("Zadatak: per-visit next-step predikcija VF_mean (proxy-MD)")
    print(f"Feature-i po koraku: {features_list}")
    print(f"Ukupan broj sekvenci (pacijent-oko): {X.shape[0]}")
    print(f"Oblik ulaza X: {X.shape} (N_uzoraka, Max_koraka, N_features)")
    print(f"Oblik izlaza y: {y.shape} (N_uzoraka, Max_koraka) - VF_mean SLEDEĆE posete po koraku")
    print(f"Oblik maske: {mask.shape} (1.0 gde y postoji, 0.0 na padding pozicijama)")
    print("Fajlovi sačuvani: X_gru.npy, y_gru.npy, mask_gru.npy, lengths_gru.npy")
    print("==========================================================")


if __name__ == "__main__":
    main()