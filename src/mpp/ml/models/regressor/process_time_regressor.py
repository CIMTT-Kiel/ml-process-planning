#standard library imports
import logging

#third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

#custom imports
from mpp.constants import VOCAB
from mpp.ml.models.regressor.trsfm_encoder_regressor import TrsfmEncoderRegressor
from mpp.ml.datasets.fabricad import Fabricad

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")


class ProcessRegressionModule(pl.LightningModule):
    """
    PyTorch Lightning Module for training a transformer-based regression model.

    This module wraps the Regressor model and handles training, validation, and optimization.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vectors.
    embed_dim : int
        Size of the embedding dimension in the transformer.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout rate.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for regularization.
    max_epochs : int
        Max number of training epochs.
    """
    def __init__(
        self,
        input_dim=32,
        embed_dim=512,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        lr=1e-4,
        weight_decay=0.01,
        max_epochs=100
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TrsfmEncoderRegressor(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        self.criterion = nn.MSELoss()
        self.max_epochs = max_epochs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.max_epochs, 
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }