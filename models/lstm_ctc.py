import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class LSTM_TemporalClassification(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=1, num_classes=109):
        super(LSTM_TemporalClassification, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.log_softmax(x)

        return x


class LSTM_TemporalClassification_PL(pl.LightningModule):
    def __init__(self, input_size=3, hidden_size=128, num_layers=1, num_classes=109, blank=0):
        super(LSTM_TemporalClassification_PL, self).__init__()

        self.model = LSTM_TemporalClassification(
            input_size, hidden_size, num_layers, num_classes
        )
        self.criterion = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        input_lengths = torch.full((y_hat.size(0),), y_hat.size(1), dtype=torch.long)
        target_lengths = torch.full((y_hat.size(0),), y.size(1), dtype=torch.long)
        loss = self.criterion(y_hat.permute(1, 0, 2), y, input_lengths, target_lengths)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = F.log_softmax(y_hat, dim=2)
        input_lengths = torch.full((y_hat.size(0),), y_hat.size(1), dtype=torch.long)
        target_lengths = torch.full((y_hat.size(0),), y.size(1), dtype=torch.long)
        loss = self.criterion(y_hat.permute(1, 0, 2), y, input_lengths, target_lengths)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        y_hat = F.log_softmax(y_hat, dim=2)
        input_lengths = torch.full((y_hat.size(0),), y_hat.size(1), dtype=torch.long)
        target_lengths = torch.full((y_hat.size(0),), y.size(1), dtype=torch.long)
        loss = self.criterion(y_hat.permute(1, 0, 2), y, input_lengths, target_lengths)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # TODO: optimize with lr_scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
        return optimizer


if __name__ == "__main__":
    model = LSTM_TemporalClassification_PL()
    # feature has shape (1, 3)
    x = torch.randn((10, 1621, 3))
    out = model(x)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")
    target = torch.randint(1, 109, (10, 100), dtype=torch.long)
    input_lengths = torch.full((10,), 1621, dtype=torch.long)
    target_lengths = torch.full((10,), 100, dtype=torch.long)
    y_hat = F.log_softmax(out, dim=2)
    loss = loss(y_hat, target, input_lengths, target_lengths)
    loss.backward()
    optim.step()
    print(out.shape)
