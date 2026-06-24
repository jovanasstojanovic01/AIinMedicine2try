import torch.nn as nn

class GlaucomaProgressionGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(GlaucomaProgressionGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.act = nn.Mish()  # Ili nn.SiLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last_step_out = out[:, -1, :] 
        activated_out = self.act(last_step_out)
        out = self.dropout_layer(activated_out)
        logits = self.fc(out)
        return logits