import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

from src import GlaucomaTemporalDataset, GlaucomaVFProgressionGRU, GlaucomaVFLoss
import config

import copy  # Trebaće nam za čuvanje najboljeg modela u memoriji

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch_x, batch_y, batch_mask, batch_lengths in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)

            preds = model(batch_x, batch_lengths)
            loss = criterion(preds, batch_y, batch_mask)

            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            all_masks.append(batch_mask.cpu().numpy())

    avg_loss = running_loss / max(total_samples, 1)

    preds_flat = np.concatenate(all_preds, axis=0).flatten()
    targets_flat = np.concatenate(all_targets, axis=0).flatten()
    masks_flat = np.concatenate(all_masks, axis=0).flatten()

    valid = masks_flat > 0.5
    mae = mean_absolute_error(targets_flat[valid], preds_flat[valid])
    rmse = root_mean_squared_error(targets_flat[valid], preds_flat[valid])
    r2 = r2_score(targets_flat[valid], preds_flat[valid])

    return avg_loss, mae, rmse, r2


def collate_with_lengths(batch):
    """Pravi lengths tenzor iz batch-a (lengths se izvodi iz maske, ne
    čuva se posebno u Dataset-u da bi se izbeglo nekonzistentno čuvanje
    dve verzije iste informacije)."""
    xs, ys, masks = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    masks = torch.stack(masks)
    lengths = masks.sum(dim=1).long().clamp(min=1)
    return xs, ys, masks, lengths


def main():
    device = config.DEVICE

    x_path = os.path.join(config.OUTPUT_DIR, "X_gru.npy")
    y_path = os.path.join(config.OUTPUT_DIR, "y_gru.npy")
    mask_path = os.path.join(config.OUTPUT_DIR, "mask_gru.npy")

    checkpoint_dir = config.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(mask_path)):
        print("[GREŠKA] Nedostaju X_gru.npy / y_gru.npy / mask_gru.npy. Pokreni create_gru_sequences.py prvo.")
        return

    full_dataset = GlaucomaTemporalDataset(x_path, y_path, mask_path)
    X_raw = full_dataset.X
    y_raw = full_dataset.y
    mask_raw = full_dataset.mask

    n_samples, max_steps, num_features = X_raw.shape
    print(f"\n================ UČITAVANJE PODATAKA ================")
    print(f"Broj sekvenci (oči): {n_samples} | Max koraka: {max_steps} | Feature-a: {num_features}")

    indices = np.arange(n_samples)
    train_idx, val_idx = train_test_split(indices, test_size=0.20, random_state=42)

    X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
    y_train, y_val = y_raw[train_idx], y_raw[val_idx]
    mask_train, mask_val = mask_raw[train_idx], mask_raw[val_idx]

    print(f"Trening sekvenci: {len(train_idx)} | Validacionih sekvenci: {len(val_idx)}")

    # StandardScaler se fituje SAMO na trening podacima (i samo na
    # validnim, ne-padding koracima), da se izbegne curenje informacija
    # iz validacionog seta u statistiku skaliranja.
    scaler = StandardScaler()
    X_train_2d = X_train_raw.reshape(-1, num_features)
    valid_rows_train = ~np.all(X_train_2d == 0, axis=1)
    scaler.fit(X_train_2d[valid_rows_train])

    X_train_scaled = scaler.transform(X_train_raw.reshape(-1, num_features)).reshape(X_train_raw.shape)
    X_val_scaled = scaler.transform(X_val_raw.reshape(-1, num_features)).reshape(X_val_raw.shape)

    train_dataset_t = list(zip(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(mask_train, dtype=torch.float32),
    ))
    val_dataset_t = list(zip(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(mask_val, dtype=torch.float32),
    ))

    train_loader = DataLoader(
        train_dataset_t, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_with_lengths
    )
    val_loader = DataLoader(
        val_dataset_t, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_with_lengths
    )

    model = GlaucomaVFProgressionGRU(
        input_size=num_features,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    ).to(device)

    criterion = GlaucomaVFLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    gru_checkpoint_path = os.path.join(checkpoint_dir, "gru_best_overall.pth")
    scaler_path = os.path.join(checkpoint_dir, "scaler.pkl")
    patience = 15
    stagnation_counter = 0
    print(f"\n--- Pokretanje treninga kroz {config.EPOCHS} epoha ---")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0
        
        for batch_x, batch_y, batch_mask, batch_lengths in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x, batch_lengths)
            loss = criterion(preds, batch_y, batch_mask)
            
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * batch_x.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Evaluacija
        val_loss, val_mae, val_rmse, val_r2 = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # === DODATO: Čuvanje najboljih težina u memoriji i reset brojača ===
            best_model_wts = copy.deepcopy(model.state_dict())
            stagnation_counter = 0
            # ==================================================================
            
            torch.save(model.state_dict(), gru_checkpoint_path)
            print(
                f"   [Novi minimum] Epoha {epoch} -> Val MSE: {val_loss:.4f} "
                f"| MAE: {val_mae:.3f} dB-ekv | RMSE: {val_rmse:.3f} | R2: {val_r2:.3f}"
            )
        # === DODATO: Ako nema poboljšanja, uvećaj brojač stagnacije ===
        else:
            stagnation_counter += 1
            print(
                f"   [Bez poboljšanja] Epoha {epoch} -> Val MSE: {val_loss:.4f} "
                f"(Stagnacija: {stagnation_counter}/{patience})"
            )
        # ==============================================================
            
        if epoch % 5 == 0 or epoch == config.EPOCHS:
            print(
                f"Epoha {epoch}/{config.EPOCHS} | Train MSE: {epoch_train_loss:.4f} "
                f"| Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.3f}"
            )

        # === DODATO: Provera uslova za Early Stopping prekid ===
        if stagnation_counter >= patience:
            print(f"\n[EARLY STOPPING] Trening prekinut u epohi {epoch} jer se Val MSE nije smanjio tokom poslednjih {patience} epoha.")
            break

    print("\n================ REZULTATI TRENINGA ================")
    print(f"Najbolji Val MSE: {best_val_loss:.4f}")
    print("======================================================")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, linestyle="--", color="blue", label="Train MSE")
    plt.plot(val_losses, linestyle="-", color="red", label="Val MSE")
    plt.title("GRU per-visit predikcija VF_mean (next-step) — kriva učenja")
    plt.xlabel("Epohe")
    plt.ylabel("MSE")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "gru_training_loss_chart.png"))
    print("Grafik krive učenja sačuvan u outputs/gru_training_loss_chart.png")

    joblib.dump(scaler, scaler_path)
    print(f"[Scaler Sačuvan] StandardScaler uspešno eksportovan u: {scaler_path}")
    print(f"[Model Sačuvan] Najbolji GRU checkpoint u: {gru_checkpoint_path}")


if __name__ == "__main__":
    main()