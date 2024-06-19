from data import InkmlDataset_PL
from models.lstm_ctc import LSTM_TemporalClassification_PL
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

if __name__ == "__main__":
    model = LSTM_TemporalClassification_PL()
    dm = InkmlDataset_PL(root_dir="dataset/crohme2019")
    logger = TensorBoardLogger("logs", name="lstm_ctc")
    
    trainer = Trainer(
        callbacks=[LearningRateMonitor(logging_interval="step"),
                   ModelCheckpoint(monitor="val_loss", save_top_k=5, mode="min"),
                   EarlyStopping(monitor="val_loss", patience=10, mode="min")],
        max_epochs=100,
        devices=[0],
        num_sanity_val_steps=0,
        fast_dev_run=True,
        log_every_n_steps=1,
        default_root_dir="checkpoint/",
        logger=logger
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)