import torch.nn as nn

class GlaucomaProgressionLoss(nn.Module):
    def __init__(self):
        super(GlaucomaProgressionLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        return self.loss_fn(preds, targets)