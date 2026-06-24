import os
import pandas as pd
import config

def main():
    grape_excel_path = os.path.join(config.DATA_DIR, "VF and clinical information.xlsx")
    unet_features_path = os.path.join(config.OUTPUT_DIR, "grape_extracted_features.csv")
    
    output_baseline_path = os.path.join(config.OUTPUT_DIR, "grape_baseline_merged.csv")
    output_followup_path = os.path.join(config.OUTPUT_DIR, "grape_followup_merged.csv")
    
    if not os.path.exists(grape_excel_path):
        print(f"[GREŠKA] Nije pronađen GRAPE Excel fajl na putanji: {grape_excel_path}")
        return
    if not os.path.exists(unet_features_path):
        print(f"[GREŠKA] Nije pronađen CSV sa UNet osobinama na putanji: {unet_features_path}")
        return

    print("-> Učitavanje podataka...")
    unet_df = pd.read_csv(unet_features_path)
    
    df_baseline = pd.read_excel(grape_excel_path, sheet_name=0)
    df_followup = pd.read_excel(grape_excel_path, sheet_name=1)
    
    print(f"-> Broj zapisa pre spajanja - Baseline: {len(df_baseline)}, Follow-up: {len(df_followup)}")

    print("-> Spajanje UNet parametara sa Sheet 1 (Baseline)...")
    baseline_merged = pd.merge(df_baseline, unet_df, on="Corresponding CFP", how="left")
    
    print("-> Spajanje UNet parametara sa Sheet 2 (Follow-up)...")
    followup_merged = pd.merge(df_followup, unet_df, on="Corresponding CFP", how="left")
    
    missing_b = baseline_merged['vCDR'].isna().sum()
    missing_f = followup_merged['vCDR'].isna().sum()
    
    if missing_b > 0:
        print(f"[UPOZORENJE] U Baseline tabeli ima {missing_b} redova koji nisu dobili UNet parametre.")
    if missing_f > 0:
        print(f"[UPOZORENJE] U Follow-up tabeli ima {missing_f} redova koji nisu dobili UNet parametre.")
        
    unet_cols = ['vCDR', 'hCDR', 'aCDR', 'Rim_Area_Pixels']
    for col in unet_cols:
        if baseline_merged[col].isna().sum() > 0:
            baseline_merged[col] = baseline_merged[col].fillna(baseline_merged[col].mean())
        if followup_merged[col].isna().sum() > 0:
            followup_merged[col] = followup_merged[col].fillna(followup_merged[col].mean())

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    baseline_merged.to_csv(output_baseline_path, index=False)
    followup_merged.to_csv(output_followup_path, index=False)
    
    print("\n================ SPAJANJE PODATAKA USPEŠNO ================")
    print(f"1. Baseline tabela sačuvana na:  {output_baseline_path}")
    print(f"2. Follow-up tabela sačuvana na: {output_followup_path}")
    print(f"Dodata obeležja: {unet_cols}")
    print("==========================================================")

if __name__ == "__main__":
    main()