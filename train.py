from data import InkmlDataset_PL
from models.lstm_ctc import LSTM_TemporalClassification_PL
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    model = LSTM_TemporalClassification_PL()
    dm = InkmlDataset_PL()
    
    trainer = Trainer(
        callbacks=[ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")],
        max_epochs=100,
        devices=1,
        num_sanity_val_steps=0,
        fast_dev_run=True,
        log_every_n_steps=1,
    )

    trainer.fit(model, dm)
    print("Done training")