import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_TemporalClassification(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=1, num_classes=109):
        super(LSTM_TemporalClassification, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

    
if __name__ == '__main__':
    model = LSTM_TemporalClassification()
    x = torch.randn((20, 20, 1))
    y, _ = model(x)