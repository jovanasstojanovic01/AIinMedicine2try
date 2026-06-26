import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class GlaucomaProgressionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GlaucomaProgressionGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x, lengths):
        
        lengths_cpu = lengths.cpu().int()
        
        packed_x = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        
        _, hn = self.gru(packed_x)
        
        last_hidden = hn[-1] 
        
        out = self.fc(last_hidden)
        return out