import logging

import mlflow
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.attention import SDPBackend, sdpa_kernel

from mpp.ml.models.regressor.trsfm_encoder_regressor import TrsfmEncoderRegressor

logger = logging.getLogger(__name__)


class ProcessRegressionModule(pl.LightningModule):
    """
    PyTorch Lightning Module für das Training des Transformer-Regressionsmodells.

    Die Zielgröße wird intern z-score-normalisiert (target_mean, target_std),
    sodass der HuberLoss immer in einer stabilen Größenordnung (~1.0) liegt.
    Metriken werden zusätzlich in absoluten Minuten geloggt (val_mae, val_rmse).

    Parameters
    ----------
    input_dim : int
        Dimensionalität der Eingabevektoren.
    embed_dim : int
        Größe des Embedding-Raums im Transformer.
    num_heads : int
        Anzahl der Attention Heads.
    num_layers : int
        Anzahl der Transformer-Encoder-Layer.
    dropout : float
        Dropout-Rate.
    lr : float
        Lernrate.
    weight_decay : float
        Weight Decay für Regularisierung.
    max_epochs : int
        Maximale Anzahl Trainings-Epochen (wird für den LR-Scheduler benötigt).
    use_scheduler : bool
        Ob CosineAnnealingLR verwendet werden soll. Im Hyperparameter-Tuning
        auf False setzen, damit kurze Trials vergleichbar bleiben.
    target_mean : float
        Mittelwert der Zielgröße aus den Trainingsdaten (für Normalisierung).
    target_std : float
        Standardabweichung der Zielgröße aus den Trainingsdaten.
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
        max_epochs=100,
        use_scheduler=True,
        target_mean=0.0,
        target_std=1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TrsfmEncoderRegressor(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        # HuberLoss: delta=1.0 in normalisiertem Raum ≙ 1 Std der Zielgröße
        # → quadratisch bei kleinen Fehlern, linear bei Ausreißern (robust)
        self.criterion = nn.HuberLoss(delta=1.0)

    def _normalize(self, y):
        return (y - self.hparams.target_mean) / self.hparams.target_std

    def _denormalize(self, y_norm):
        return y_norm * self.hparams.target_std + self.hparams.target_mean

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), self._normalize(y))
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds_norm = self(x)
        loss = self.criterion(preds_norm, self._normalize(y))

        preds_abs = self._denormalize(preds_norm)
        mae  = (preds_abs - y).abs().mean()
        rmse = ((preds_abs - y) ** 2).mean().sqrt()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae",  mae,  on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds_norm = self(x)
        loss = self.criterion(preds_norm, self._normalize(y))

        preds_abs = self._denormalize(preds_norm)
        mae  = (preds_abs - y).abs().mean()
        rmse = ((preds_abs - y) ** 2).mean().sqrt()

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_mae",  mae,  on_step=False, on_epoch=True)
        self.log("test_rmse", rmse, on_step=False, on_epoch=True)

    def on_train_start(self):
        """Loggt den tatsächlichen Flash-Attention-Status als MLflow-Parameter."""
        device = next(self.parameters()).device
        precision = self.trainer.precision

        on_cuda = device.type == "cuda"
        is_low_precision = any(p in str(precision) for p in ("16", "bf16"))

        flash_active = False
        if on_cuda and is_low_precision:
            try:
                nhead = self.hparams.num_heads
                head_dim = self.hparams.embed_dim // nhead
                dummy = torch.randn(1, nhead, 4, head_dim, device=device, dtype=torch.bfloat16)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    torch.nn.functional.scaled_dot_product_attention(dummy, dummy, dummy)
                flash_active = True
            except Exception:
                flash_active = False

        mlflow.log_params({
            "flash_attention_active": flash_active,
            "training_device": str(device),
            "training_precision": str(precision),
            "batch_size": self.trainer.train_dataloader.batch_size,
            "target_mean": self.hparams.target_mean,
            "target_std":  self.hparams.target_std,
        })

        status = "AKTIV" if flash_active else "INAKTIV"
        logger.info(f"Flash Attention: {status}  (device={device}, precision={precision})")
        logger.info(f"Ziel-Normalisierung: mean={self.hparams.target_mean:.2f} min, "
                    f"std={self.hparams.target_std:.2f} min")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if not self.hparams.use_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
