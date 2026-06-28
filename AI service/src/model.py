import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GlaucomaVFProgressionGRU(nn.Module):
    """
    GRU za PER-VISIT predikciju: na osnovu istorije poseta do koraka t,
    predviđa VF_mean (proxy-MD) na poseti t+1.

    Ovo je regresija na nivou SVAKOG koraka u sekvenci (many-to-many),
    za razliku od prethodne verzije koja je čitala samo poslednje skriveno
    stanje (hn[-1]) i klasifikovala CELO oko jednim fiksnim labelom
    (PLR2/PLR3/MD progression status). Ovde svaka poseta u nizu nosi
    sopstveni target — VF_mean SLEDEĆE posete — što ima klinički smisao
    na nivou pojedinačnog pregleda, ne samo na nivou celog oka.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GlaucomaVFProgressionGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout_layer = nn.Dropout(dropout)
        # Regresija: jedan skalar (VF_mean sledeće posete) po koraku.
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        """
        x: [batch, max_steps, input_size]  (right-padded)
        lengths: [batch]  broj VALIDNIH koraka po uzorku

        Vraća: [batch, max_steps]  predviđeni VF_mean za SVAKI korak
        (interpretacija: izlaz na poziciji t je predikcija za posetu t+1).
        Pozicije iza 'lengths' su nedefinisane i moraju biti maskirane
        pri računanju loss-a (vidi mask_gru.npy / GlaucomaVFLoss).
        """
        lengths_cpu = lengths.cpu().int()

        packed_x = pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed_x)

        # Vraćamo na puni (padded) oblik da bismo dobili izlaz PO KORAKU,
        # ne samo poslednje skriveno stanje.
        out, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )

        out = self.dropout_layer(out)
        preds = self.fc(out)  # [batch, max_steps, 1]
        return preds.squeeze(-1)  # [batch, max_steps]