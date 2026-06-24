import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Primena sigmoid funkcije da prebacimo logits u verovatnoće (0 do 1)
        preds = torch.sigmoid(preds)
        
        # Sravnjujemo tenzore (flatten) da olakšamo računanje po pikselima
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedDiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super(CombinedDiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)

    def forward(self, preds, targets):
        """
        preds: Izlaz iz modela oblika [Batch, Kanali, H, W] - sirovi logits
        targets: Prave maske iz dataseta istog oblika
        """
        # Računamo gubitak za oba kanala (Kanal 0: Disk, Kanal 1: Kup)
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        
        # Kombinovana težinska suma
        total_loss = (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)
        return total_loss