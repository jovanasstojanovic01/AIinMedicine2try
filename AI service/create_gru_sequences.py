import os
import numpy as np
import pandas as pd
import config

def main():
    baseline_path = os.path.join(config.OUTPUT_DIR, "grape_baseline_merged.xlsx")
    followup_path = os.path.join(config.OUTPUT_DIR, "grape_followup_merged.xlsx")
    
    output_x_path = os.path.join(config.OUTPUT_DIR, "X_gru.npy")
    output_y_path = os.path.join(config.OUTPUT_DIR, "y_gru.npy")
    output_lengths_path = os.path.join(config.OUTPUT_DIR, "lengths_gru.npy")

    if not os.path.exists(baseline_path) or not os.path.exists(followup_path):
        print("[GREŠKA] Prvo moraš pokrenuti prethodni korak (merge_grape_data.py)!")
        return

    print("-> Učitavanje unifikovanih tabela...")
    df_b = pd.read_excel(baseline_path)
    df_f = pd.read_excel(followup_path)

    df_b["Visit Number"] = 0
    
    # 1. ISPRAVKA MAPIRANJA: Pravilno imenujemo sva tri flegova na osnovu strukture Excela
    baseline_rename_map = {
        "Progression Status": "PLR2",  # Prva spojena ćelija
        "Unnamed: 8": "PLR3",          # Druga spojena ćelija
        "Unnamed: 9": "MD_flag"        # Treća spojena ćelija
    }
    df_b.rename(columns=baseline_rename_map, inplace=True)
    
    features_list = ["IOP", "vCDR", "hCDR", "aCDR", "Rim_Area_Pixels"]
    id_cols = ["Subject Number", "Laterality", "Visit Number"]
    
    # Tri ciljne labele koje model treba da predvidi
    target_labels = ["PLR2", "PLR3", "MD_flag"]
    
    df_b_sub = df_b[id_cols + features_list + target_labels]
    df_f_sub = df_f[id_cols + features_list].copy()
    
    # 2. PROPASTAVALJANJE SVA TRI FLEGA KROZ CELU ISTORIJU PACIJENTA
    # Pravimo rečnike za sva tri statusa iz baseline-a
    plr2_map = df_b.set_index(["Subject Number", "Laterality"])["PLR2"].to_dict()
    plr3_map = df_b.set_index(["Subject Number", "Laterality"])["PLR3"].to_dict()
    md_map = df_b.set_index(["Subject Number", "Laterality"])["MD_flag"].to_dict()
    
    # Preslikavamo ih na followup preglede
    df_f_sub["PLR2"] = df_f_sub.set_index(["Subject Number", "Laterality"]).index.map(plr2_map)
    df_f_sub["PLR3"] = df_f_sub.set_index(["Subject Number", "Laterality"]).index.map(plr3_map)
    df_f_sub["MD_flag"] = df_f_sub.set_index(["Subject Number", "Laterality"]).index.map(md_map)

    # Spajamo sve u hronološki DataFrame
    df_all = pd.concat([df_b_sub, df_f_sub], axis=0, ignore_index=True)
    df_all = df_all.sort_values(by=["Subject Number", "Laterality", "Visit Number"]).reset_index(drop=True)

    print("-> Kreiranje hronoloških sekvenci po pacijentima...")
    X_sequences = []
    y_labels = []
    lengths_list = [] 
    
    max_visits = int(df_all["Visit Number"].max()) + 1
    num_features = len(features_list)
    
    grouped = df_all.groupby(["Subject Number", "Laterality"])
    
    for (_, _), group in grouped:
        group = group.sort_values(by="Visit Number")
        
        current_features = group[features_list].values
        
        # 3. KREIRANJE MULTI-LABEL Y NIZA: Uzimamo sva tri statusa odjednom [PLR2, PLR3, MD]
        label_plr2 = group["PLR2"].iloc[0]
        label_plr3 = group["PLR3"].iloc[0]
        label_md   = group["MD_flag"].iloc[0]
        
        # Smeštamo ih kao vektor sa tri elementa
        three_labels = [label_plr2, label_plr3, label_md]
        
        padded_features = np.zeros((max_visits, num_features), dtype=np.float32)
        actual_steps = current_features.shape[0] 
        
        padded_features[-actual_steps:] = current_features
        
        X_sequences.append(padded_features)
        y_labels.append(three_labels) # y sada čuva nizove od po 3 elementa
        lengths_list.append(actual_steps)

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.float32) # Oblik će biti: (broj_očiju, 3)
    lengths = np.array(lengths_list, dtype=np.int32) 

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.save(output_x_path, X)
    np.save(output_y_path, y)
    np.save(output_lengths_path, lengths)

    print("\n================ SEKVENCIRANJE ZAVRŠENO ================")
    print(f"Ukupan broj sekvenci (pacijenti-oči): {X.shape[0]}")
    print(f"Oblik ulaza X: {X.shape} (N_uzoraka, Max_poseta, N_features)")
    print(f"Oblik izlaza y: {y.shape} (N_uzoraka, 3_labele)")
    print(f"Fajlovi uspešno sačuvani.")
    print("========================================================")

if __name__ == "__main__":
    main()