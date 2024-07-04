import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchaudio.models.decoder import ctc_decoder, cuda_ctc_decoder
import numpy as np
from .utils.edit import TokenEditDistance
from .utils.loss import RelativePositionLoss

# Constants
ALPHA = 0.1  # Relative Position Loss weight


class LSTM_TemporalClassification(nn.Module):
    def __init__(
        self, input_size=4, hidden_size=256, num_layers=2, num_classes=109, **kwargs
    ):
        super(LSTM_TemporalClassification, self).__init__()

        bidirectional = kwargs.get("bidirectional", False)

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.log_softmax(x)

        return x


class LSTM_TemporalClassification_PL(pl.LightningModule):
    def __init__(
        self,
        input_size=4,
        hidden_size=256,
        num_layers=2,
        num_classes=109,
        blank=0,
        **kwargs
    ):
        super(LSTM_TemporalClassification_PL, self).__init__()

        bidirectional = True

        self.save_hyperparameters(logger=False)
        self.model = LSTM_TemporalClassification(
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            bidirectional=bidirectional,
        )
        self.criterion = nn.CTCLoss(blank=blank, zero_infinity=True, reduction="mean")
        self.relation_criterion = RelativePositionLoss()
        self.vocab = {
            "": 0,
            "-": 1,
            "\\times": 2,
            "\\{": 3,
            "\\beta": 4,
            "m": 5,
            "Above": 6,
            "E": 7,
            "\\infty": 8,
            "\\forall": 9,
            "\\cos": 10,
            "8": 11,
            ")": 12,
            "/": 13,
            "\\sum": 14,
            "n": 15,
            "\\pi": 16,
            "\\geq": 17,
            "C": 18,
            "a": 19,
            "\\mu": 20,
            "S": 21,
            "]": 22,
            "R": 23,
            "\\gt": 24,
            "Sup": 25,
            "x": 26,
            "p": 27,
            "\\ldots": 28,
            "\\int": 29,
            "\\sqrt": 30,
            "f": 31,
            "Right": 32,
            "k": 33,
            "\\log": 34,
            "\\leq": 35,
            "j": 36,
            "w": 37,
            "7": 38,
            "y": 39,
            "\\exists": 40,
            "d": 41,
            "[": 42,
            "q": 43,
            "\\div": 44,
            "NoRel": 45,
            "\\phi": 46,
            "1": 47,
            "g": 48,
            "X": 49,
            "\\in": 50,
            "\\gamma": 51,
            "\\prime": 52,
            "4": 53,
            "\\pm": 54,
            "T": 55,
            "F": 56,
            "N": 57,
            "\\lt": 58,
            "o": 59,
            "u": 60,
            "h": 61,
            "s": 62,
            "6": 63,
            "c": 64,
            "(": 65,
            "A": 66,
            "!": 67,
            "P": 68,
            "L": 69,
            "COMMA": 70,
            "i": 71,
            "b": 72,
            "t": 73,
            "+": 74,
            "\\neq": 75,
            "9": 76,
            "3": 77,
            "G": 78,
            ".": 79,
            "e": 80,
            "M": 81,
            "r": 82,
            "\\sin": 83,
            "\\lim": 84,
            "\\lambda": 85,
            "I": 86,
            "\\rightarrow": 87,
            "Inside": 88,
            "\\sigma": 89,
            "V": 90,
            "\\theta": 91,
            "l": 92,
            "=": 93,
            "\\tan": 94,
            "z": 95,
            "2": 96,
            "H": 97,
            "0": 98,
            "5": 99,
            "Below": 100,
            "|": 101,
            "\\Delta": 102,
            "\\alpha": 103,
            "B": 104,
            "Y": 105,
            "v": 106,
            "\\}": 107,
            "Sub": 108,
        }
        self.relation = ["Above", "Below", "Inside", "NoRel", "Right", "Sub", "Sup"]
        self.relation_idx = [6, 100, 88, 45, 32, 108, 25]
        self.decoder = cuda_ctc_decoder(tokens=list(self.vocab.keys()))
        self.metric = TokenEditDistance()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, in_len, target_len = batch
        batch_size = x.size(0)
        y_hat = self.model(x)
        input_lengths = torch.tensor(in_len).cuda()
        target_lengths = torch.tensor(target_len).cuda()

        ctc_loss = self.criterion(y_hat.permute(1, 0, 2), y, input_lengths, target_lengths)
        rel_loss = self.relation_criterion(y_hat, x[:, :, 3])  # pen_up

        total_loss = ctc_loss + (ALPHA * rel_loss)

        self.log("train_CTCloss", ctc_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_RelLoss", rel_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y, in_len, target_len = batch
        batch_size = x.size(0)
        y_hat = self.model(x)
        input_lengths = torch.tensor(in_len).cuda()
        target_lengths = torch.tensor(target_len).cuda()

        ctc_loss = self.criterion(y_hat.permute(1, 0, 2), y, input_lengths, target_lengths)
        rel_loss = self.relation_criterion(y_hat, x[:, :, 3])  # pen_up

        total_loss = ctc_loss + (ALPHA * rel_loss)

        keys = list(self.vocab.keys())
        decoded_output = self.decoder(y_hat, input_lengths.to(torch.int32))

        output_str_list = [" ".join(output[0].words) for output in decoded_output]
        target_str_list = [" ".join([keys[i] for i in target.cpu().numpy()]).strip() for target in y]
        edit_distance = self.metric(output_str_list, target_str_list)

        output_relation = [" ".join([x for x in output[0].words if x in self.relation]) for output in decoded_output]
        target_relation = [" ".join([keys[i] for i in target.cpu().numpy() if i in self.relation_idx]).strip() for target in y]
        relation_edit_distance = self.metric(output_relation, target_relation)

        output_symbol = [" ".join([x for x in output[0].words if x not in self.relation]) for output in decoded_output]
        target_symbol = [" ".join([keys[i] for i in target.cpu().numpy() if i not in self.relation_idx]).strip() for target in y]
        symbol_edit_distance = self.metric(output_symbol, target_symbol)

        wer = edit_distance / np.array([len(target.split()) for target in target_str_list]).mean()

        self.log("val_CTCloss", ctc_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_RelLoss", rel_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        self.log("edit_distance", edit_distance, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("relation_edit_distance", relation_edit_distance, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        self.log("symbol_edit_distance", symbol_edit_distance, on_epoch=True, prog_bar=False, logger=True, batch_size=batch_size)
        self.log("total_wer", wer, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return total_loss

    def test_step(self, batch, batch_idx):
        x, y, in_len, target_len = batch
        batch_size = x.size(0)
        y_hat = self.model(x)
        input_lengths = torch.tensor(in_len).cuda()
        target_lengths = torch.tensor(target_len).cuda()

        ctc_loss = self.criterion(y_hat.permute(1, 0, 2), y, input_lengths, target_lengths)
        rel_loss = self.relation_criterion(y_hat, x[:, :, 3])

        total_loss = ctc_loss + (ALPHA * rel_loss)

        keys = list(self.vocab.keys())
        decoded_output = self.decoder(y_hat, input_lengths.to(torch.int32))

        output_str_list = [" ".join(output[0].words) for output in decoded_output]
        target_str_list = [" ".join([keys[i] for i in target.cpu().numpy()]).strip() for target in y]
        edit_distance = self.metric(output_str_list, target_str_list)

        output_relation = [" ".join([x for x in output[0].words if x in self.relation]) for output in decoded_output]
        target_relation = [" ".join([keys[i] for i in target.cpu().numpy() if i in self.relation_idx]).strip() for target in y]
        relation_edit_distance = self.metric(output_relation, target_relation)

        output_symbol = [" ".join([x for x in output[0].words if x not in self.relation]) for output in decoded_output]
        target_symbol = [" ".join([keys[i] for i in target.cpu().numpy() if i not in self.relation_idx]).strip() for target in y]
        symbol_edit_distance = self.metric(output_symbol, target_symbol)

        wer = edit_distance / np.array([len(target.split()) for target in target_str_list]).mean()

        self.log("test_CTCloss", ctc_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("test_RelLoss", rel_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("test_loss", total_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        self.log("edit_distance", edit_distance, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("relation_edit_distance", relation_edit_distance, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("symbol_edit_distance", symbol_edit_distance, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("total_wer", wer, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # TODO: optimize with lr_scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
        # return optimizer


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
