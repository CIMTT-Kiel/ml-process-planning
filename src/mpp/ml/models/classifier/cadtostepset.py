import logging

import mlflow
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.attention import SDPBackend, sdpa_kernel

from mpp.constants import VOCAB
from mpp.ml.models.classifier.multilabel_classifier import MultilabelTransformerEncoderModule

logger = logging.getLogger(__name__)


class ProcessClassificationTrsfmEncoderModule(pl.LightningModule):
    """
    PyTorch Lightning Module für das Training des Multi-Label-Klassifikationsmodells.

    Bildet einen Satz von Eingabevektoren (CAD-Daten) auf eine Menge von
    Prozessschritten ab (Multi-Label, ungeordnet).

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
    num_classes : int
        Anzahl der Prozessschritt-Klassen (ohne START, STOP, PAD).
    dropout : float
        Dropout-Rate.
    lr : float
        Lernrate.
    weight_decay : float
        Weight Decay für Regularisierung.
    threshold : float
        Sigmoid-Schwellenwert für binäre Vorhersage.
    max_epochs : int
        Maximale Anzahl Trainings-Epochen (wird für den LR-Scheduler benötigt).
    use_scheduler : bool
        Ob CosineAnnealingLR verwendet werden soll. Im Hyperparameter-Tuning
        auf False setzen, damit kurze Trials vergleichbar bleiben.
    pos_weight : torch.Tensor or None
        Klassengewichte für BCEWithLogitsLoss (neg_count / pos_count pro Klasse).
        Pflicht bei unbalancierten Klassen (z.B. schleifen/kontrollieren ~8%).
    """

    def __init__(
        self,
        input_dim=32,
        embed_dim=512,
        num_heads=4,
        num_layers=2,
        num_classes=len(VOCAB) - 3,  # ohne STOP, PAD, START
        dropout=0.3,
        lr=1e-4,
        weight_decay=0.01,
        threshold=0.5,
        max_epochs=50,
        use_scheduler=True,
        pos_weight=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])

        self.model = MultilabelTransformerEncoderModule(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            threshold=threshold,
        )
        # pos_weight wird in on_train_start auf das richtige Device verschoben
        self._pos_weight = pos_weight
        self.criterion = nn.BCEWithLogitsLoss()

        # Akkumulation für epochenweisen Macro-F1
        self._val_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        assert y.dim() == 2, f"Expected y shape (B, C), got {y.shape}"
        assert y.dtype == torch.float, f"Expected y dtype float, got {y.dtype}"

        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits) > self.hparams.threshold
        acc = (preds == y.bool()).float().mean()

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        assert y.dim() == 2, f"Expected y shape (B, C), got {y.shape}"
        assert y.dtype == torch.float, f"Expected y dtype float, got {y.dtype}"

        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits) > self.hparams.threshold
        acc = (preds == y.bool()).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        self._val_outputs.append((preds.detach().cpu(), y.bool().cpu()))

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            return

        preds = torch.cat([o[0] for o in self._val_outputs])    # [N, C]
        targets = torch.cat([o[1] for o in self._val_outputs])  # [N, C]
        self._val_outputs.clear()

        f1_per_class = []
        for c in range(preds.size(1)):
            tp = ((preds[:, c]) & (targets[:, c])).sum().float()
            fp = ((preds[:, c]) & (~targets[:, c])).sum().float()
            fn = ((~preds[:, c]) & (targets[:, c])).sum().float()
            denom = 2 * tp + fp + fn
            f1_per_class.append((2 * tp / denom).item() if denom > 0 else 0.0)

        macro_f1 = sum(f1_per_class) / len(f1_per_class)
        self.log("val_f1_macro", macro_f1, prog_bar=True)

    def on_train_start(self):
        """Verschiebt pos_weight auf das Trainings-Device und loggt Flash-Attention-Status."""
        device = next(self.parameters()).device
        precision = self.trainer.precision

        # pos_weight auf richtiges Device bringen und Criterion aktualisieren
        if self._pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=self._pos_weight.to(device)
            )

        # Flash Attention prüfen
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
        })

        status = "AKTIV" if flash_active else "INAKTIV"
        logger.info(f"Flash Attention: {status}  (device={device}, precision={precision})")

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


if __name__ == "__main__":
    model = ProcessClassificationTrsfmEncoderModule(
        lr=0.001,
        embed_dim=512,
        num_layers=4,
        dropout=0.1,
        weight_decay=0.01,
        max_epochs=10,
    )
