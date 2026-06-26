import torch
import torch.nn as nn

class GlaucomaProgressionLoss(nn.Module):
    def __init__(self):
        super(GlaucomaProgressionLoss, self).__init__()
        weight_tensor = torch.tensor([0.9], dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    def forward(self, preds, targets):
        if targets.dim() == 1 and preds.dim() == 2:
            targets = targets.unsqueeze(-1)
        return self.loss_fn(preds, targets)