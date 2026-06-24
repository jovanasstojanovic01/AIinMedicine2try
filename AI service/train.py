import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

from src import GlaucomaTemporalDataset, GlaucomaProgressionLSTM, GlaucomaProgressionLoss
import config
from xgboost import XGBClassifier

def train_single_fold(fold, model, train_loader, val_loader, criterion, optimizer, epochs, device, checkpoint_dir, global_best_val_loss, xgb_model):
    fold_best_val_loss = float('inf')
    fold_train_losses = []
    fold_val_losses = []
    
    best_fold_preds = None
    best_fold_targets = None
    
    overall_lstm_path = os.path.join(checkpoint_dir, "lstm_best_overall.pth")
    overall_xgb_path = os.path.join(checkpoint_dir, "xgb_best_overall.pkl")

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)  # Pretvara [16, 1] u [16]
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * batch_x.size(0)
            total_train += batch_y.size(0)
            
        epoch_train_loss = running_train_loss / total_train
        
        # --- VALIDACIJA: ENSEMBLE (LSTM + XGBOOST) ---
        model.eval()
        all_lstm_preds = []
        all_targets = []
        all_xgb_preds = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x_dev = batch_x.to(device)
                logits = model(batch_x_dev).squeeze(-1)
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_lstm_preds.extend(probs)
                all_targets.extend(batch_y.numpy())
                
                batch_x_xgb = batch_x.numpy().reshape(batch_x.size(0), -1)
                xgb_p = xgb_model.predict_proba(batch_x_xgb)[:, 1]
                all_xgb_preds.extend(xgb_p)
        
        all_lstm_preds = np.array(all_lstm_preds)
        all_xgb_preds = np.array(all_xgb_preds)
        all_targets = np.array(all_targets)
        
        ensemble_preds = (all_lstm_preds + all_xgb_preds) / 2.0
        epoch_val_loss = log_loss(all_targets, ensemble_preds, labels=[0, 1])
        
        fold_train_losses.append(epoch_train_loss)
        fold_val_losses.append(epoch_val_loss)
        
        if epoch_val_loss < fold_best_val_loss:
            fold_best_val_loss = epoch_val_loss
            best_fold_preds = ensemble_preds.copy()
            best_fold_targets = all_targets.copy()
            
        if epoch_val_loss < global_best_val_loss:
            global_best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), overall_lstm_path)
            joblib.dump(xgb_model, overall_xgb_path)
            print(f"   [NOVI REKORD PIPELINE-A] Epoha {epoch} u Foldu {fold} -> Novi najbolji Hibridni Loss: {global_best_val_loss:.4f}!")
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"   Epoha [{epoch}/{epochs}] -> Train Loss: {epoch_train_loss:.4f} | Kombinovani Val Loss: {epoch_val_loss:.4f}")
            
    print(f"-> [Fold {fold} Završen] Najbolji Hibridni Val Loss u ovom foldu: {fold_best_val_loss:.4f}")
    return fold_train_losses, fold_val_losses, fold_best_val_loss, global_best_val_loss, best_fold_preds, best_fold_targets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x_path = os.path.join(config.OUTPUT_DIR, "X_lstm.npy")
    y_path = os.path.join(config.OUTPUT_DIR, "y_lstm.npy")
    checkpoint_dir = config.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    full_dataset = GlaucomaTemporalDataset(x_path, y_path)
    X_raw = full_dataset.X
    y_data = full_dataset.y
    
    n_samples, t_steps, num_features = X_raw.shape
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"\n================ START K-FOLD CROSS VALIDACIJE ({n_splits} Folds) ================")
    
    all_folds_train_losses = []
    all_folds_val_losses = []
    best_scores_per_fold = []
    global_best_val_loss = float('inf')
    
    fold_accuracies, fold_aucs, fold_sensitivities, fold_specificities = [], [], [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_data), 1):
        print(f"\n--- TRENIRANJE FOLD-A {fold}/{n_splits} ---")
        
        X_train_raw = X_raw[train_idx].copy()
        y_train_orig = y_data[train_idx].copy()
        X_val_raw = X_raw[val_idx].copy()
        y_val = y_data[val_idx]
        
        # --- SKALIRANJE SA MASKIRANJEM PADDING NULA ---
        scaler = StandardScaler()
        X_train_2d = X_train_raw.reshape(-1, num_features)
        
        # Izbacujemo nule iz proračuna srednje vrednosti i varijanse
        mask = ~(np.all(X_train_2d == 0, axis=1))
        scaler.fit(X_train_2d[mask])
        
        # Skaliranje kompletnih skupova
        X_train_scaled = scaler.transform(X_train_2d).reshape(X_train_raw.shape)
        X_val_scaled = scaler.transform(X_val_raw.reshape(-1, num_features)).reshape(X_val_raw.shape)
        
        # --- ČISTA AUGMENTACIJA PODATAKA ---
        print(f"   [Augmentacija] Kreiranje sintetičkih pacijenata za trening...")
        noise = np.random.normal(0, 0.05, size=X_train_scaled.shape)
        X_train_augmented = X_train_scaled + noise
        y_train_augmented = y_train_orig.copy()
        
        X_train_final = np.concatenate([X_train_scaled, X_train_augmented], axis=0)
        y_train_final = np.concatenate([y_train_orig, y_train_augmented], axis=0)
        print(f"   [Augmentacija] Trening skup proširen sa {len(X_train_scaled)} na {len(X_train_final)} uzoraka.")
        
        # Datasets & Loaders
        train_tensor_x = torch.tensor(X_train_final, dtype=torch.float32)
        train_tensor_y = torch.tensor(y_train_final, dtype=torch.float32)
        train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
        
        val_tensor_x = torch.tensor(X_val_scaled, dtype=torch.float32)
        val_tensor_y = torch.tensor(y_val, dtype=torch.float32)
        val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # --- XGBOOST TRENING ---
        print(f"   [XGBoost] Fitovanje modela sa skaliranim i augmentisanim podacima...")
        X_train_xgb = X_train_final.reshape(X_train_final.shape[0], -1)
        
        xgb_model = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss'
        )
        xgb_model.fit(X_train_xgb, y_train_final, verbose=False)
        
        # --- LSTM INICIJALIZACIJA ---
        model = GlaucomaProgressionLSTM(
            input_size=num_features, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT
        ).to(device)
        
        criterion = GlaucomaProgressionLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        
        t_losses, v_losses, fold_best_loss, global_best_val_loss, f_preds, f_targets = train_single_fold(
            fold, model, train_loader, val_loader, criterion, optimizer, config.EPOCHS, device, checkpoint_dir, global_best_val_loss, xgb_model
        )
        
        all_folds_train_losses.append(t_losses)
        all_folds_val_losses.append(v_losses)
        best_scores_per_fold.append(fold_best_loss)
        
        # --- METRIKE PO FOLDOVIMA ---
        f_preds_bin = (f_preds >= 0.5).astype(int)
        fold_acc = accuracy_score(f_targets, f_preds_bin)
        fold_auc = roc_auc_score(f_targets, f_preds)
        
        tn, fp, fn, tp = confusion_matrix(f_targets, f_preds_bin).ravel()
        fold_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        fold_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        fold_accuracies.append(fold_acc)
        fold_aucs.append(fold_auc)
        fold_sensitivities.append(fold_sens)
        fold_specificities.append(fold_spec)
        
    print("\n================ EVALUACIJA UNIKRSNE VALIDACIJE ================")
    print(f"Prosečan najbolji Kombinovani (Hibridni) Loss kroz sve foldove: {np.mean(best_scores_per_fold):.4f}")
    print(f"Prosečna Tačnost (Accuracy) kroz sve foldove:          {np.mean(fold_accuracies)*100:.2f}%")
    print(f"Prosečan ROC-AUC Score kroz sve foldove:               {np.mean(fold_aucs):.4f}")
    print(f"Prosečna Osetljivost (Sensitivity/Recall):            {np.mean(fold_sensitivities)*100:.2f}%")
    print(f"Prosečna Specifičnost (Specificity):                  {np.mean(fold_specificities)*100:.2f}%")
    print("==================================================================")
    
    plt.figure(figsize=(10, 6))
    for fold in range(n_splits):
        plt.plot(all_folds_train_losses[fold], linestyle='--', alpha=0.5, label=f"Fold {fold+1} Train Loss (LSTM)")
        plt.plot(all_folds_val_losses[fold], linestyle='-', alpha=0.9, label=f"Fold {fold+1} Kombinovani Val Loss")
    plt.title("Hibridni Model (LSTM + XGBoost) - Maskiran Padding")
    plt.xlabel("Epohe")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "lstm_kfold_loss_chart.png"))

if __name__ == "__main__":
    main()