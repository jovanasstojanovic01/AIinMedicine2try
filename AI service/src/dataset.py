import numpy as np
import torch
from torch.utils.data import Dataset


class GlaucomaTemporalDataset(Dataset):
    """
    Učitava X (feature sekvence), y (VF_mean sledeće posete po koraku) i
    mask (1.0 gde y važi, 0.0 na padding pozicijama).

    Ranija verzija je učitavala samo X i y (jedan label po sekvenci za
    PLR2/PLR3/MD). Pošto je novi zadatak per-step regresija, dataset mora
    da prosledi i masku, jer poslednja (validna) poseta svakog oka nema
    "sledeću" posetu i ne treba da bude paddovana nulama kao da je 0.0
    legitimna meta vrednost.
    """

    def __init__(self, x_path, y_path, mask_path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        self.mask = np.load(mask_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        mask = torch.tensor(self.mask[idx], dtype=torch.float32)
        return seq, target, mask