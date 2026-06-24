import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split

from src import GlaucomaTemporalDataset, GlaucomaProgressionGRU, GlaucomaProgressionLoss
import config
from xgboost import XGBClassifier

def train_single_fold(fold, model, train_loader, val_loader, criterion, optimizer, epochs, device, checkpoint_dir, global_best_val_loss, xgb_model):
    fold_best_val_loss = float('inf')
    fold_train_losses = []
    fold_val_losses = []
    
    best_fold_preds = None
    best_fold_targets = None
    
    overall_gru_path = os.path.join(checkpoint_dir, "gru_best_overall.pth")
    overall_xgb_path = os.path.join(checkpoint_dir, "xgb_best_overall.pkl")

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x).squeeze(-1)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_train_loss += loss.item() * batch_x.size(0)
            total_train += batch_y.size(0)
            
        epoch_train_loss = running_train_loss / total_train
        
        model.eval()
        all_gru_preds = []
        all_targets = []
        all_xgb_preds = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x_dev = batch_x.to(device)
                logits = model(batch_x_dev).squeeze(-1)
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_gru_preds.extend(probs)
                all_targets.extend(batch_y.numpy())
                
                batch_x_xgb = batch_x.numpy().reshape(batch_x.size(0), -1)
                xgb_p = xgb_model.predict_proba(batch_x_xgb)[:, 1]
                all_xgb_preds.extend(xgb_p)
        
        all_gru_preds = np.array(all_gru_preds)
        all_xgb_preds = np.array(all_xgb_preds)
        all_targets = np.array(all_targets)
        
        ensemble_preds = (0.4 * all_gru_preds) + (0.6 * all_xgb_preds)
        epoch_val_loss = log_loss(all_targets, ensemble_preds, labels=[0, 1])
        
        fold_train_losses.append(epoch_train_loss)
        fold_val_losses.append(epoch_val_loss)
        
        if epoch_val_loss < fold_best_val_loss:
            fold_best_val_loss = epoch_val_loss
            best_fold_preds = ensemble_preds.copy()
            best_fold_targets = all_targets.copy()
            
        if epoch_val_loss < global_best_val_loss:
            global_best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), overall_gru_path)
            joblib.dump(xgb_model, overall_xgb_path)
            print(f"   [Novi minimum] Epoha {epoch} -> Kombinovani Loss: {global_best_val_loss:.4f}")
            
    return fold_train_losses, fold_val_losses, fold_best_val_loss, global_best_val_loss, best_fold_preds, best_fold_targets

def main():
    device = config.DEVICE
    
    x_path = os.path.join(config.OUTPUT_DIR, "X_gru.npy")
    y_path = os.path.join(config.OUTPUT_DIR, "y_gru.npy")
    checkpoint_dir = config.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    full_dataset = GlaucomaTemporalDataset(x_path, y_path)
    X_raw = full_dataset.X
    y_data = full_dataset.y
    
    _, _, num_features = X_raw.shape
    
    X_train_raw, X_val_raw, y_train_orig, y_val = train_test_split(
        X_raw, y_data, test_size=0.20, stratify=y_data, random_state=42
    )
    
    print(f"\n================ START JEDNOSTRUKE EVALUACIJE ================")
    print(f"Trening uzoraka: {len(X_train_raw)} | Validacionih uzoraka: {len(X_val_raw)}")

    scaler = StandardScaler()
    X_train_2d = X_train_raw.reshape(-1, num_features)
    mask = ~(np.all(X_train_2d == 0, axis=1))
    scaler.fit(X_train_2d[mask])
    
    X_train_scaled = scaler.transform(X_train_2d).reshape(X_train_raw.shape)
    X_val_scaled = scaler.transform(X_val_raw.reshape(-1, num_features)).reshape(X_val_raw.shape)
   
    print(f"   [Augmentacija] Kreiranje sintetičkih pacijenata...")
    noise = np.random.normal(0, 0.05, size=X_train_scaled.shape)
    X_train_augmented = X_train_scaled + noise
    y_train_augmented = y_train_orig.copy()
    
    X_train_final = np.concatenate([X_train_scaled, X_train_augmented], axis=0)
    y_train_final = np.concatenate([y_train_orig, y_train_augmented], axis=0)
    
    train_tensor_x = torch.tensor(X_train_final, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train_final, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
    
    val_tensor_x = torch.tensor(X_val_scaled, dtype=torch.float32)
    val_tensor_y = torch.tensor(y_val, dtype=torch.float32)
    val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"   [XGBoost] Fitovanje modela...")
    X_train_xgb = X_train_final.reshape(X_train_final.shape[0], -1)
    xgb_model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train_xgb, y_train_final, verbose=False)
    
    model = GlaucomaProgressionGRU(
        input_size=num_features, hidden_size=config.HIDDEN_SIZE, num_layers=config.NUM_LAYERS, dropout=config.DROPOUT
    ).to(device)
    
    criterion = GlaucomaProgressionLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    
    global_best_val_loss = float('inf')
    
    print(f"\n--- Pokretanje treninga kroz {config.EPOCHS} epoha ---")
    t_losses, v_losses, best_loss, _, f_preds, f_targets = train_single_fold(
        1, model, train_loader, val_loader, criterion, optimizer, config.EPOCHS, device, checkpoint_dir, global_best_val_loss, xgb_model
    )
    
    f_preds_bin = (f_preds >= 0.5).astype(int)
    acc = accuracy_score(f_targets, f_preds_bin)
    auc = roc_auc_score(f_targets, f_preds)
    
    print("\n================ REZULTATI JEDNOSTRUKE PODELE ================")
    print(f"Najbolji Kombinovani Val Loss: {best_loss:.4f}")
    print(f"Tačnost (Accuracy):           {acc*100:.2f}%")
    print(f"ROC-AUC Score:                 {auc:.4f}")
    print("===============================================================")
    
    plt.figure(figsize=(8, 5))
    plt.plot(t_losses, linestyle='--', color='blue', label="Train Loss (GRU)")
    plt.plot(v_losses, linestyle='-', color='red', label="Kombinovani Val Loss")
    plt.title("Jednostruka podela (Train/Val Split) - Ponašanje modela")
    plt.xlabel("Epohe")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "single_split_loss_chart.png"))

if __name__ == "__main__":
    main()