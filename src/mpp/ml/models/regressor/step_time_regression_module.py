"""PyTorch-Lightning-Modul für die schrittweise Fertigungszeitvorhersage.

Implementiert:

* **Teacher Forcing** (Training): GT-Zeiten als Decoder-Input.
* **Scheduled Sampling** (optional): zunehmend eigene Vorhersagen als Input.
* **Zwei Trainingsphasen**:

  1. Phase 1 (Epochen 0 … ``freeze_encoder_epochs-1``): Encoder eingefroren,
     nur Decoder wird trainiert.
  2. Phase 2 (ab Epoche ``freeze_encoder_epochs``): Encoder wird mit niedrigerer
     LR (``lr * encoder_lr_factor``) mittrainiert.

* **Verlust**: Huber-Loss pro Schritt + gewichteter Consistency-Loss.
* **Normalisierung**: Z-Score auf Schrittzeit-Ebene (mean/std aus Trainingsdaten).

Verwendung
----------
::

    # Modell von Grund auf neu:
    module = StepTimeRegressionModule(embed_dim=128, ...)

    # Mit vortrainiertem Encoder:
    module = StepTimeRegressionModule.from_pretrained_encoder(
        ckpt_path="path/to/process_time.ckpt",
        embed_dim=128,
        ...
    )
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn.attention import SDPBackend, sdpa_kernel

from mpp.constants import VOCAB
from mpp.ml.models.regressor.step_time_decoder import (
    StepTimeDecoder,
    TrsfmEncoderStepTimeModel,
)
from mpp.ml.models.regressor.trsfm_encoder_regressor import TrsfmEncoderRegressor

logger = logging.getLogger(__name__)


class StepTimeRegressionModule(pl.LightningModule):
    """PyTorch-Lightning-Wrapper für das Encoder-Decoder-Modell zur
    schrittweisen Zeitvorhersage.

    Parameters
    ----------
    input_dim : int
        Dimensionalität der Eingabevektoren im Vector-Set (Standard: 32).
    embed_dim : int
        Embedding-Dimension (Encoder *und* Decoder). Standard: 128.
    num_heads : int
        Anzahl der Attention-Köpfe. Standard: 8.
    num_encoder_layers : int
        Anzahl der Transformer-Encoder-Layer. Standard: 4.
    num_decoder_layers : int
        Anzahl der Transformer-Decoder-Layer. Standard: 4.
    dropout : float
        Dropout-Rate. Standard: 0.1.
    max_seq_len : int
        Maximale Sequenzlänge (für gelernte Positional-Embeddings). Standard: 12.
    lr : float
        Lernrate für Decoder (und Encoder in Phase 2). Standard: 1e-4.
    encoder_lr_factor : float
        Faktor, mit dem ``lr`` für den Encoder in Phase 2 multipliziert wird.
        Standard: 0.1 → Encoder-LR = 0.1 × lr.
    weight_decay : float
        Weight-Decay. Standard: 0.01.
    max_epochs : int
        Maximale Trainings-Epochen (für LR-Scheduler). Standard: 100.
    use_scheduler : bool
        CosineAnnealingLR verwenden?  Im Hyperparameter-Tuning auf ``False``
        setzen, damit kurze Trials vergleichbar bleiben.
    target_mean : float
        Mittelwert der per-Schritt-Zeiten aus dem Trainingsset (Minuten).
    target_std : float
        Standardabweichung der per-Schritt-Zeiten (Minuten).
    lambda_consistency : float
        Gewichtungsfaktor λ für den Consistency-Loss.  Standard: 0.1.
    freeze_encoder_epochs : int
        Anzahl Epochen, in denen der Encoder eingefroren bleibt (Phase 1).
        Standard: 20.
    scheduled_sampling : bool
        Scheduled Sampling aktivieren?  Standard: ``False`` (reines Teacher
        Forcing).
    scheduled_sampling_rate : float
        Anteil der Trainings-Batches, bei denen in jeder Epoche Scheduled
        Sampling statt Teacher Forcing verwendet wird.  Standard: 0.5.
    """

    def __init__(
        self,
        input_dim: int = 32,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 12,
        lr: float = 1e-4,
        encoder_lr_factor: float = 0.1,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        use_scheduler: bool = True,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        lambda_consistency: float = 0.1,
        freeze_encoder_epochs: int = 20,
        scheduled_sampling: bool = False,
        scheduled_sampling_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        encoder = TrsfmEncoderRegressor(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )
        decoder = StepTimeDecoder(
            vocab_size=len(VOCAB),
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.model = TrsfmEncoderStepTimeModel(encoder, decoder)

        # Huber-Loss elementweise (reduction='none') für PAD-Maskierung
        self.huber = nn.HuberLoss(delta=1.0, reduction="none")

    # ------------------------------------------------------------------
    # Normalisierungs-Hilfsmethoden
    # ------------------------------------------------------------------

    def _normalize(self, y: torch.Tensor) -> torch.Tensor:
        """Z-Score-Normalisierung in den normierten Raum."""
        return (y - self.hparams.target_mean) / self.hparams.target_std

    def _denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Z-Score-Denormalisierung zurück in Minuten."""
        return y_norm * self.hparams.target_std + self.hparams.target_mean

    # ------------------------------------------------------------------
    # Vorwärtsdurchlauf
    # ------------------------------------------------------------------

    def forward(
        self,
        vecset: torch.Tensor,
        step_tokens: torch.Tensor,
        prev_times: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-Forcing-Forward (normalisierter Raum)."""
        return self.model(vecset, step_tokens, prev_times)

    # ------------------------------------------------------------------
    # Scheduled Sampling (autoregressiver Forward im Trainings-Step)
    # ------------------------------------------------------------------

    def _scheduled_forward(
        self,
        vecset: torch.Tensor,
        step_tokens: torch.Tensor,
        step_times_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Scheduled-Sampling-Forward: Decoder-Input wird Schritt für Schritt
        aus den *eigenen* Vorhersagen aufgebaut statt aus GT-Zeiten.

        NOTE: Dieser Modus erfordert seq_len sequentielle Decoder-Aufrufe und
        ist daher langsamer als Teacher Forcing.  Für Produktions-Training
        empfiehlt sich ein Curriculum, das die Rate über die Epochen steigert;
        hier wird ein konstanter ``scheduled_sampling_rate``-Anteil verwendet.

        Parameters
        ----------
        vecset : torch.Tensor
            Geometrie-Embeddings ``[B, set_size, 32]``.
        step_tokens : torch.Tensor
            Token-IDs ``[B, seq_len]``.
        step_times_norm : torch.Tensor
            Normalisierte GT-Zeiten ``[B, seq_len]`` (nicht verwendet,
            nur zur Signatur-Kompatibilität behalten).

        Returns
        -------
        torch.Tensor
            Normalisierte Zeitvorhersagen ``[B, seq_len]``.
        """
        B, seq_len = step_tokens.shape
        memory = self.model.encoder.encode(vecset)             # [B, set_size, E]

        all_pred: list[torch.Tensor] = []

        for i in range(seq_len):
            # prev_times: [0, pred_{0}, pred_{1}, …, pred_{i-1}]
            prev_so_far = torch.zeros(B, i + 1, device=vecset.device)
            if i > 0:
                prev_so_far[:, 1:] = torch.stack(all_pred, dim=1)  # [B, i]

            tokens_so_far = step_tokens[:, : i + 1]
            out = self.model.decoder(memory, tokens_so_far, prev_so_far)
            t_i = out[:, -1]                                   # [B]

            # PAD-Position → 0
            t_i = t_i.masked_fill(tokens_so_far[:, -1] == VOCAB["PAD"], 0.0)
            all_pred.append(t_i)

        return torch.stack(all_pred, dim=1)                    # [B, seq_len]

    # ------------------------------------------------------------------
    # Loss-Berechnung
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        pred_times_norm: torch.Tensor,
        step_times_norm: torch.Tensor,
        step_tokens: torch.Tensor,
        total_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Berechnet Huber-Loss + Consistency-Loss.

        Parameters
        ----------
        pred_times_norm : torch.Tensor  [B, seq_len]
            Vorhersagen im normierten Raum.
        step_times_norm : torch.Tensor  [B, seq_len]
            GT-Zeiten im normierten Raum (PAD-Positionen sind 0.0).
        step_tokens : torch.Tensor      [B, seq_len]
            Token-IDs (zur PAD-Maskierung).
        total_time : torch.Tensor       [B]
            GT-Gesamtzeit in Minuten (Summe der gefilterten Schritte).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(gesamt_loss, huber_loss, consistency_loss)``
        """
        pad_mask = step_tokens == VOCAB["PAD"]                 # [B, seq_len]
        valid_count = (~pad_mask).float().sum().clamp(min=1)

        # Huber-Loss: elementweise, PAD-Positionen ausgeblendet (normierter Raum)
        huber_elem = self.huber(pred_times_norm, step_times_norm)  # [B, seq_len]
        huber_loss = huber_elem.masked_fill(pad_mask, 0.0).sum() / valid_count

        # Consistency-Loss: im normierten Raum berechnen, damit die Skala mit dem
        # Huber-Loss übereinstimmt.
        #
        # Für n valide Schritte gilt:
        #   sum(norm_i) = (total_time - n * mean) / std
        # Deshalb total_time direkt in den normierten Summen-Raum transformieren,
        # ohne Denormalisierung – dadurch bleibt der Loss in [normierte Einheit]²
        # und λ ist ein sinnvoller Kompromissparameter.
        n_valid_per_sample = (~pad_mask).float().sum(dim=-1).clamp(min=1)      # [B]
        total_time_norm_sum = (
            (total_time - n_valid_per_sample * self.hparams.target_mean)
            / self.hparams.target_std
        )                                                                        # [B]
        pred_total_norm = pred_times_norm.masked_fill(pad_mask, 0.0).sum(dim=-1)  # [B]
        consistency_loss = F.mse_loss(pred_total_norm, total_time_norm_sum)

        total_loss = (
            huber_loss
            + self.hparams.lambda_consistency * consistency_loss
        )
        return total_loss, huber_loss, consistency_loss

    # ------------------------------------------------------------------
    # Lightning-Steps
    # ------------------------------------------------------------------

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        vecset, step_tokens, step_times, total_time = batch

        step_times_norm = self._normalize(step_times)

        use_scheduled = (
            self.hparams.scheduled_sampling
            and torch.rand(1, device=self.device).item()
            < self.hparams.scheduled_sampling_rate
        )

        if use_scheduled:
            pred_times_norm = self._scheduled_forward(
                vecset, step_tokens, step_times_norm
            )
        else:
            # Teacher Forcing: prev_times = GT-Zeiten um 1 nach rechts verschoben
            prev_times = F.pad(
                step_times_norm[:, :-1], (1, 0), value=0.0
            )  # [B, seq_len]
            pred_times_norm = self.model(vecset, step_tokens, prev_times)

        loss, huber, consistency = self._compute_loss(
            pred_times_norm, step_times_norm, step_tokens, total_time
        )

        self.log("train_loss",        loss,        on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_huber",       huber,       on_step=False, on_epoch=True)
        self.log("train_consistency", consistency, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> None:
        vecset, step_tokens, step_times, total_time = batch
        step_times_norm = self._normalize(step_times)

        # Validation immer mit Teacher Forcing (reproduzierbar + schnell)
        prev_times = F.pad(step_times_norm[:, :-1], (1, 0), value=0.0)
        pred_times_norm = self.model(vecset, step_tokens, prev_times)

        loss, huber, consistency = self._compute_loss(
            pred_times_norm, step_times_norm, step_tokens, total_time
        )

        # Metriken in absoluten Minuten
        pad_mask = step_tokens == VOCAB["PAD"]
        pred_abs = self._denormalize(pred_times_norm).masked_fill(pad_mask, 0.0)
        gt_abs   = step_times.masked_fill(pad_mask, 0.0)

        valid = (~pad_mask).float().sum().clamp(min=1)
        mae  = (pred_abs - gt_abs).abs().sum() / valid
        rmse = ((pred_abs - gt_abs) ** 2).sum().div(valid).sqrt()

        self.log("val_loss",        loss,        on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_huber",       huber,       on_step=False, on_epoch=True)
        self.log("val_consistency", consistency, on_step=False, on_epoch=True)
        self.log("val_mae",         mae,         on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse",        rmse,        on_step=False, on_epoch=True)

    def test_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> None:
        vecset, step_tokens, step_times, total_time = batch
        step_times_norm = self._normalize(step_times)

        prev_times = F.pad(step_times_norm[:, :-1], (1, 0), value=0.0)
        pred_times_norm = self.model(vecset, step_tokens, prev_times)

        loss, huber, consistency = self._compute_loss(
            pred_times_norm, step_times_norm, step_tokens, total_time
        )

        pad_mask = step_tokens == VOCAB["PAD"]
        pred_abs = self._denormalize(pred_times_norm).masked_fill(pad_mask, 0.0)
        gt_abs   = step_times.masked_fill(pad_mask, 0.0)
        valid = (~pad_mask).float().sum().clamp(min=1)
        mae  = (pred_abs - gt_abs).abs().sum() / valid
        rmse = ((pred_abs - gt_abs) ** 2).sum().div(valid).sqrt()

        self.log("test_loss",        loss,        on_step=False, on_epoch=True)
        self.log("test_huber",       huber,       on_step=False, on_epoch=True)
        self.log("test_consistency", consistency, on_step=False, on_epoch=True)
        self.log("test_mae",         mae,         on_step=False, on_epoch=True)
        self.log("test_rmse",        rmse,        on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Zwei-Phasen-Training: Encoder einfrieren / auftauen
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """Schaltet den Encoder in Epoche ``freeze_encoder_epochs`` auf
        trainierbar um (Phase 2).

        In Phase 1 (Epochen 0 … freeze_encoder_epochs-1) sind alle Encoder-
        Parameter ``requires_grad=False``.  Der AdamW-Optimizer ignoriert
        sie damit, was einer kompletten Einfrierung entspricht.

        Ab Phase 2 werden sie auf ``requires_grad=True`` gesetzt; der zweite
        Parameter-Gruppen im Optimizer (Encoder, niedrige LR) wird dann aktiv.
        """
        freeze = self.hparams.freeze_encoder_epochs
        encoder_params = list(self.model.encoder.parameters())

        if self.current_epoch < freeze:
            for p in encoder_params:
                p.requires_grad = False
        elif self.current_epoch == freeze:
            logger.info(
                f"Epoche {self.current_epoch}: Phase 2 – Encoder wird aufgetaut "
                f"(LR = {self.hparams.lr * self.hparams.encoder_lr_factor:.2e})."
            )
            for p in encoder_params:
                p.requires_grad = True

    # ------------------------------------------------------------------
    # Flash-Attention-Logging
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        """Loggt Flash-Attention-Status und Normalisierungsparameter via MLflow."""
        device    = next(self.parameters()).device
        precision = self.trainer.precision

        on_cuda        = device.type == "cuda"
        is_low_prec    = any(p in str(precision) for p in ("16", "bf16"))
        flash_active   = False

        if on_cuda and is_low_prec:
            try:
                nh  = self.hparams.num_heads
                hd  = self.hparams.embed_dim // nh
                dummy = torch.randn(1, nh, 4, hd, device=device, dtype=torch.bfloat16)
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    F.scaled_dot_product_attention(dummy, dummy, dummy)
                flash_active = True
            except Exception:
                flash_active = False

        mlflow.log_params({
            "flash_attention_active": flash_active,
            "training_device":        str(device),
            "training_precision":     str(precision),
            "batch_size":             self.trainer.train_dataloader.batch_size,
            "target_mean":            self.hparams.target_mean,
            "target_std":             self.hparams.target_std,
            "freeze_encoder_epochs":  self.hparams.freeze_encoder_epochs,
            "lambda_consistency":     self.hparams.lambda_consistency,
        })

        status = "AKTIV" if flash_active else "INAKTIV"
        logger.info(f"Flash Attention: {status}  (device={device}, precision={precision})")
        logger.info(
            f"Normalisierung: mean={self.hparams.target_mean:.2f} min, "
            f"std={self.hparams.target_std:.2f} min"
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        """Zwei Parametergruppen mit differentieller Lernrate.

        * Decoder (+ Regressor-Kopf des Encoders): ``lr``
        * Encoder (embedding + TransformerEncoder): ``lr * encoder_lr_factor``

        In Phase 1 sind Encoder-Gewichte eingefroren (``requires_grad=False``),
        daher werden sie vom Optimizer ignoriert, auch wenn die Gruppe existiert.
        Ab Phase 2 sind sie aktiv und erhalten die niedrigere LR automatisch.
        """
        encoder_params = list(self.model.encoder.parameters())
        decoder_params = list(self.model.decoder.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.hparams.lr * self.hparams.encoder_lr_factor},
                {"params": decoder_params, "lr": self.hparams.lr},
            ],
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

    # ------------------------------------------------------------------
    # Klassenmethode: vortrainierter Encoder laden
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_encoder(
        cls,
        ckpt_path: str,
        **kwargs: Any,
    ) -> "StepTimeRegressionModule":
        """Erstellt ein neues Modul und lädt die Encoder-Gewichte aus einem
        vorhandenen ``ProcessRegressionModule``-Checkpoint.

        Nur ``embedding``- und ``encoder``-Gewichte werden übernommen;
        der ``regressor``-Kopf des alten Modells wird ignoriert, da er in
        der neuen Architektur keine Rolle spielt.

        NOTE: ``embed_dim``, ``num_heads`` und ``num_encoder_layers`` in
        ``kwargs`` müssen mit den Werten des geladenen Checkpoints
        übereinstimmen, da sonst ein Shape-Mismatch entsteht.

        Parameters
        ----------
        ckpt_path : str
            Pfad zur ``*.ckpt``-Datei des vortrainierten ``ProcessRegressionModule``.
        **kwargs
            Weitere Initialisierungsargumente für :class:`StepTimeRegressionModule`
            (z. B. ``lr``, ``lambda_consistency``, ``target_mean``, …).

        Returns
        -------
        StepTimeRegressionModule
            Neues Modul mit vorgeladenen Encoder-Gewichten.

        Examples
        --------
        >>> module = StepTimeRegressionModule.from_pretrained_encoder(
        ...     "checkpoints/best_model/time-regression/model.ckpt",
        ...     embed_dim=128,
        ...     num_encoder_layers=4,
        ...     lr=1e-4,
        ...     target_mean=42.0,
        ...     target_std=18.5,
        ... )
        """
        # Lazy import vermeidet zirkuläre Abhängigkeit
        from mpp.ml.models.regressor.process_time_regressor import ProcessRegressionModule

        pretrained = ProcessRegressionModule.load_from_checkpoint(ckpt_path)
        pretrained_model: TrsfmEncoderRegressor = pretrained.model

        module = cls(**kwargs)

        # Embedding- und Encoder-Gewichte explizit übertragen
        module.model.encoder.embedding.load_state_dict(
            pretrained_model.embedding.state_dict()
        )
        module.model.encoder.encoder.load_state_dict(
            pretrained_model.encoder.state_dict()
        )

        logger.info(f"Encoder-Gewichte aus '{ckpt_path}' geladen.")
        return module
