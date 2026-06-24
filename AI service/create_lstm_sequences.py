import os
import numpy as np
import pandas as pd
import config

# from sklearn.preprocessing import StandardScaler

def main():
    baseline_path = os.path.join(config.OUTPUT_DIR, "grape_baseline_merged.csv")
    followup_path = os.path.join(config.OUTPUT_DIR, "grape_followup_merged.csv")
    
    output_x_path = os.path.join(config.OUTPUT_DIR, "X_lstm.npy")
    output_y_path = os.path.join(config.OUTPUT_DIR, "y_lstm.npy")

    if not os.path.exists(baseline_path) or not os.path.exists(followup_path):
        print("[GREŠKA] Prvo moraš pokrenuti prethodni korak (merge_grape_data.py)!")
        return

    print("-> Učitavanje unifikovanih tabela...")
    df_b = pd.read_csv(baseline_path)
    df_f = pd.read_csv(followup_path)

    df_b["Visit Number"] = 0
    
    features_list = ["IOP", "vCDR", "hCDR", "aCDR", "Rim_Area_Pixels"]
    
    id_cols = ["Subject Number", "Laterality", "Visit Number"]
    
    df_b_sub = df_b[id_cols + features_list + ["Progression Status"]]
    df_f_sub = df_f[id_cols + features_list]
    
    progression_map = df_b.set_index(["Subject Number", "Laterality"])["Progression Status"].to_dict()
    df_f_sub["Progression Status"] = df_f_sub.set_index(["Subject Number", "Laterality"]).index.map(progression_map)

    df_all = pd.concat([df_b_sub, df_f_sub], axis=0, ignore_index=True)
    
    df_all = df_all.sort_values(by=["Subject Number", "Laterality", "Visit Number"]).reset_index(drop=True)

    print("-> Skaliranje kliničkih obeležja pomoću StandardScaler-a...")
    
    # scaler = StandardScaler()
    # df_all[features_list] = scaler.fit_transform(df_all[features_list])

    print("-> Kreiranje hronoloških sekvenci po pacijentima...")
    X_sequences = []
    y_labels = []
    
    max_visits = int(df_all["Visit Number"].max()) + 1
    num_features = len(features_list)
    
    grouped = df_all.groupby(["Subject Number", "Laterality"])
    
    for (_, _), group in grouped:
        group = group.sort_values(by="Visit Number")
        
        current_features = group[features_list].values
        
        label = group["Progression Status"].iloc[0]
        
        padded_features = np.zeros((max_visits, num_features), dtype=np.float32)
        actual_steps = current_features.shape[0]
        
        padded_features[-actual_steps:] = current_features
        
        X_sequences.append(padded_features)
        y_labels.append(label)

    X = np.array(X_sequences, dtype=np.float32)
    y = np.array(y_labels, dtype=np.float32)

    os.makedirs("outputs", exist_ok=True)
    np.save(output_x_path, X)
    np.save(output_y_path, y)

    print("\n================ SEKVENCIRANJE ZAVRŠENO ================")
    print(f"Ukupan broj sekvenci (pacijenti-oči): {X.shape[0]}")
    print(f"Maksimalan broj vremenskih koraka:    {X.shape[1]}")
    print(f"Broj kliničkih obeležja po koraku:    {X.shape[2]}")
    print(f"Konačni oblik ulaza X za LSTM:        {X.shape} -> [Batch, Time_Steps, Features]")
    print(f"Konačni oblik izlaza y za LSTM:       {y.shape} -> [Batch]")
    print(f"Fajlovi uspešno sačuvani u 'outputs/' folderu.")
    print("========================================================")

if __name__ == "__main__":
    main()