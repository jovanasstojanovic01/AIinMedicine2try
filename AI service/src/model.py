import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class GlaucomaProgressionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes=3):
        super(GlaucomaProgressionGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        # ISPRAVKA: Umesto 1, linearni sloj sada mapira skriveno stanje na 3 izlaza (PLR2, PLR3, MD)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths):
        lengths_cpu = lengths.cpu().int()
        
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        
        _, hn = self.gru(packed_x)
        
        last_hidden = hn[-1] 
        
        # Izlaz ovde ima dimenziju [batch_size, 3] (sirovi logiti za sva 3 flega)
        out = self.fc(last_hidden)
        
        return out