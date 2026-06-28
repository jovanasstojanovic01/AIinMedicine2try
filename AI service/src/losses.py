import torch
import torch.nn as nn


class GlaucomaVFLoss(nn.Module):
    """
    Maskirani MSE loss za per-visit regresiju VF_mean (proxy-MD).

    Zamena za prethodni GlaucomaProgressionLoss, koji je koristio
    BCEWithLogitsLoss sa jednim skalarnim pos_weight broadcast-ovanim na
    3 binarne klase (PLR2/PLR3/MD) — neispravno jer su klase imale
    različitu raspodelu pozitivnih/negativnih primera, a dobijale su
    identičan ponder. Pošto je novi zadatak regresija jednog kontinuelnog
    cilja po koraku, BCE više nije relevantan; potreban je MSE koji
    ignoriše padding pozicije preko maske.
    """

    def __init__(self):
        super(GlaucomaVFLoss, self).__init__()

    def forward(self, preds, targets, mask):
        """
        preds, targets, mask: [batch, max_steps]
        mask je 1.0 na validnim (ne-padding) pozicijama, 0.0 inače.
        """
        squared_error = (preds - targets) ** 2
        masked_error = squared_error * mask

        # Prosek samo preko validnih pozicija (izbegavamo deljenje sa
        # brojem koji uključuje padding nule).
        denom = mask.sum().clamp(min=1.0)
        return masked_error.sum() / denom