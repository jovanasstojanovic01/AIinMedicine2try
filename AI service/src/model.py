import torch.nn as nn

class GlaucomaProgressionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(GlaucomaProgressionLSTM, self).__init__()
        
        # LSTM sa više slojeva MORA imati dropout između slojeva
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dropout pre potpuno povezanog sloja (sprečava overfitting)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # out: [Batch, Time_Steps, Hidden_Size]
        # U forward funkciji LSTM modela:
        out, (hn, cn) = self.lstm(x)

        # hn ima oblik: [num_layers, batch_size, hidden_size]
        # Uzimamo poslednji sloj neuronske mreže:
        last_hidden = hn[-1] 

        out = self.dropout_layer(last_hidden)
        logits = self.fc(out)
        # out, (hn, cn) = self.lstm(x)
        
        # # Uzimamo samo izlaz iz POSLEDNJEG vremenskog koraka (poslednje kontrole)
        # last_step_out = out[:, -1, :]
        
        # # Primenjujemo dropout i linearizaciju
        # out = self.dropout_layer(last_step_out)
        # logits = self.fc(out).squeeze(-1)  # Vraća [Batch]
        return logits