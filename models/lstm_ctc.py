import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LSTM_TemporalClassification(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=1, num_classes=109):
        super(LSTM_TemporalClassification, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


class LSTM_TemporalClassification_PL(pl.LightningModule):
    def __init__(self, input_size=3, hidden_size=128, num_layers=1, num_classes=109):
        super(LSTM_TemporalClassification_PL, self).__init__()

        self.model = LSTM_TemporalClassification(
            input_size, hidden_size, num_layers, num_classes
        )
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = F.log_softmax(y_hat, dim=2)
        input_lengths = torch.full((y_hat.size(0),), y_hat.size(1), dtype=torch.long)
        target_lengths = torch.full((y_hat.size(0),), y.size(1), dtype=torch.long)
        loss = self.criterion(y_hat, y, input_lengths, target_lengths)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = F.log_softmax(y_hat, dim=2)
        input_lengths = torch.full((y_hat.size(0),), y_hat.size(1), dtype=torch.long)
        target_lengths = torch.full((y_hat.size(0),), y.size(1), dtype=torch.long)
        loss = self.criterion(y_hat, y, input_lengths, target_lengths)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = F.log_softmax(y_hat, dim=2)
        input_lengths = torch.full((y_hat.size(0),), y_hat.size(1), dtype=torch.long)
        target_lengths = torch.full((y_hat.size(0),), y.size(1), dtype=torch.long)
        loss = self.criterion(y_hat, y, input_lengths, target_lengths)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    model = LSTM_TemporalClassification_PL()
    # feature has shape (1, 3)
    x = torch.randn((10, 1621, 3))
    out = model(x)
    print(out.shape)
