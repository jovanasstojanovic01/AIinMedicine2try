import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Uvoz naših lokalnih modula
from .. import config
from src import RefugeDataset, train_transforms, val_test_transforms, RefugeUNet, CombinedDiceBCELoss

def train_epoch(model, loader, optimizer, loss_fn, scaler, device):
    """Izvršava jednu epohu treninga nad celim trening setom"""
    model.train()
    running_loss = 0.0
    
    # tqdm pravi vizuelni progress bar u terminalu
    loop = tqdm(loader, desc="[Training]", leave=False)
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)
        
        # Resetovanje gradijenata
        optimizer.zero_grad()
        
        # Mixed Precision (ubrzava trening na novijim grafičkim kartama)
        with torch.amp.autocast(device_type='cuda' if 'cuda' in device else 'cpu'):
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
        # Backward pass i optimizacija
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return running_loss / len(loader)

def validate_epoch(model, loader, loss_fn, device):
    """Evaluira model na validacionom setu"""
    model.eval()
    running_loss = 0.0
    
    loop = tqdm(loader, desc="[Validation]", leave=False)
    with torch.no_grad():
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
    return running_loss / len(loader)

def main():
    # 1. Osiguraj da folderi za izlaze postoje
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print(f"Pokretanje treninga na uređaju: {config.DEVICE}")
    
    # 2. Inicijalizacija podataka
    train_dataset = RefugeDataset(root_dir=config.DATA_DIR, split='train', transforms=train_transforms)
    val_dataset = RefugeDataset(root_dir=config.DATA_DIR, split='val', transforms=val_test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    # 3. Inicijalizacija modela, loss funkcije i optimizatora
    model = RefugeUNet(in_channels=3, out_channels=2).to(config.DEVICE)
    loss_fn = CombinedDiceBCELoss(bce_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    # Gradscaler za Mixed Precision
    scaler = torch.amp.GradScaler('cuda' if 'cuda' in config.DEVICE else 'cpu')
    
    # Pracenje istorije radi crtanja grafika
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 4. Glavna petlja kroz epohe
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoha {epoch}/{config.EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, config.DEVICE)
        val_loss = validate_epoch(model, val_loader, loss_fn, config.DEVICE)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"-> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Čuvanje najboljeg modela (Best Checkpoint)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"*** Sačuvan novi najbolji model sa Val Loss: {val_loss:.4f} ***")
            
    # 5. Crtanje i čuvanje grafika sa loss krivama
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, config.EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, config.EPOCHS + 1), val_losses, label="Val Loss")
    plt.xlabel("Epohe")
    plt.ylabel("Loss")
    plt.title("Trening i Validacija - Gubitak kroz epohe")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "loss_chart.png"))
    plt.close()
    print("\nTrening završen! Grafik 'loss_chart.png' je sačuvan u outputs/ folderu.")

if __name__ == "__main__":
    main()