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
from create_gru_sequences import FEATURES_LIST
import config


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

    # --- Popunjavanje NaN (PRE skaliranja) ---
    # merge_grape_data.py namerno OSTAVLJA NaN u vCDR/hCDR/aCDR/Rim_Area_Pixels
    # kada UNet ekstrakcija ne pokrije neku sliku, i compute_vf_mean takođe
    # može vratiti NaN ako su sve VF vrednosti te posete -1/NaN. Ako se ti
    # NaN-ovi ne uklone PRE poziva scaler.fit/transform, propagiraju se kroz
    # CEO scaler (mean_/scale_ postaju NaN), pa StandardScaler.transform
    # vraća NaN za SVE redove, ne samo one koji su originalno imali NaN —
    # to je uzrok "Input contains NaN" greške pri treningu.
    #
    # Popunjavamo medianom (robusnija na outliere od mean), računatom
    # ISKLJUČIVO iz validnih (ne-padding) koraka TRENING dela — ne iz
    # validacionog dela — da ne unesemo data leakage.
    X_train_2d = X_train_raw.reshape(-1, num_features)
    valid_rows_train = ~np.all(X_train_2d == 0, axis=1)

    train_medians = np.nanmedian(X_train_2d[valid_rows_train], axis=0)
    if np.isnan(train_medians).any():
        # Ako je ceo feature NaN u trening delu (ekstremno retko), 0.0 je
        # neutralan fallback (isti kao padding vrednost).
        train_medians = np.nan_to_num(train_medians, nan=0.0)
        print("[UPOZORENJE] Bar jedan feature je bio NaN za SVE validne trening redove — popunjen sa 0.0.")

    n_nan_train = np.isnan(X_train_raw).sum()
    n_nan_val = np.isnan(X_val_raw).sum()
    if n_nan_train > 0 or n_nan_val > 0:
        print(f"[INFO] Pronađeno NaN vrednosti — trening: {n_nan_train}, validacija: {n_nan_val}. Popunjavam medianom trening skupa: {train_medians}")

    def fill_nan_with_train_median(X):
        X_filled = X.copy()
        for f_idx in range(num_features):
            feat_slice = X_filled[:, :, f_idx]
            nan_mask = np.isnan(feat_slice)
            feat_slice[nan_mask] = train_medians[f_idx]
        return X_filled

    X_train_raw = fill_nan_with_train_median(X_train_raw)
    X_val_raw = fill_nan_with_train_median(X_val_raw)

    # Isto važi za y (VF_mean target) — ako je target NaN, mask na toj
    # poziciji treba da bude 0 (već je tako po konstrukciji u
    # create_gru_sequences.py ako je VF_mean NaN propagiran), ali za
    # svaki slučaj NaN u y zamenjujemo sa 0.0 i FORSIRAMO mask=0 tu, da
    # loss sigurno ne vidi NaN target.
    nan_y_train = np.isnan(y_train)
    nan_y_val = np.isnan(y_val)
    if nan_y_train.any() or nan_y_val.any():
        print(f"[UPOZORENJE] Pronađeno NaN u target (y) vrednostima — trening: {nan_y_train.sum()}, validacija: {nan_y_val.sum()}. Maskiram te pozicije i nuliram target.")
    y_train = np.nan_to_num(y_train, nan=0.0)
    y_val = np.nan_to_num(y_val, nan=0.0)
    mask_train = mask_train.copy()
    mask_val = mask_val.copy()
    mask_train[nan_y_train] = 0.0
    mask_val[nan_y_val] = 0.0

    # StandardScaler se fituje SAMO na trening podacima (i samo na
    # validnim, ne-padding koracima), da se izbegne curenje informacija
    # iz validacionog seta u statistiku skaliranja.
    #
    # NAPOMENA: 'has_cfp' je binarni indikator (0.0/1.0), ne kontinuelna
    # merna vrednost — skaliranje (mean/std) bi ga nepotrebno transformisalo
    # u neke druge brojeve (npr. -1.8 / 0.4) bez ikakve koristi, i otežalo
    # bi čitanje/debug. Izdvajamo ga PRE fit/transform i vraćamo nazad
    # nepromenjenog nakon skaliranja ostalih feature-a.
    has_cfp_idx = FEATURES_LIST.index("has_cfp")
    scale_mask = np.array([i != has_cfp_idx for i in range(num_features)])

    scaler = StandardScaler()
    X_train_2d = X_train_raw.reshape(-1, num_features)
    valid_rows_train = ~np.all(X_train_2d == 0, axis=1)
    scaler.fit(X_train_2d[valid_rows_train][:, scale_mask])

    def scale_keep_flag(X_raw):
        X_2d = X_raw.reshape(-1, num_features)
        X_scaled_2d = X_2d.copy()
        X_scaled_2d[:, scale_mask] = scaler.transform(X_2d[:, scale_mask])
        return X_scaled_2d.reshape(X_raw.shape)

    X_train_scaled = scale_keep_flag(X_train_raw)
    X_val_scaled = scale_keep_flag(X_val_raw)

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

    print(f"\n--- Pokretanje treninga kroz {config.EPOCHS} epoha ---")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_train_loss = 0.0
        total_train = 0

        for batch_x, batch_y, batch_mask, batch_lengths in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad()
            preds = model(batch_x, batch_lengths)
            loss = criterion(preds, batch_y, batch_mask)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss += loss.item() * batch_x.size(0)
            total_train += batch_x.size(0)

        epoch_train_loss = running_train_loss / max(total_train, 1)
        val_loss, val_mae, val_rmse, val_r2 = evaluate(model, val_loader, criterion, device)

        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), gru_checkpoint_path)
            print(
                f"   [Novi minimum] Epoha {epoch} -> Val MSE: {val_loss:.4f} "
                f"| MAE: {val_mae:.3f} dB-ekv | RMSE: {val_rmse:.3f} | R2: {val_r2:.3f}"
            )

        if epoch % 5 == 0 or epoch == config.EPOCHS:
            print(
                f"Epoha {epoch}/{config.EPOCHS} | Train MSE: {epoch_train_loss:.4f} "
                f"| Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.3f}"
            )

    # Učitavamo NAJBOLJI checkpoint (ne poslednju epohu) da finalni
    # izveštaj odgovara modelu koji je stvarno sačuvan na disku.
    model.load_state_dict(torch.load(gru_checkpoint_path, map_location=device))
    final_val_loss, final_mae, final_rmse, final_r2 = evaluate(model, val_loader, criterion, device)

    print("\n================ REZULTATI TRENINGA (najbolji checkpoint) ================")
    print(f"Val MSE:  {final_val_loss:.4f}")
    print(f"Val MAE:  {final_mae:.3f}  (proseč. greška u jedinicama VF_mean)")
    print(f"Val RMSE: {final_rmse:.3f}")
    print(f"Val R²:   {final_r2:.3f}  (1.0 = perfektno, 0.0 = isto kao predikcija prosekom, <0 = lošije od proseka)")
    print("============================================================================")

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