"""PyTorch-Lightning-Modul für die MTL-Schritt-Zeit-und-Kosten-Vorhersage.

Implementiert:

* **Teacher Forcing** (Training): GT-Zeiten/Kosten als Decoder-Input.
* **Kendall-MTL-Loss** (Kendall et al. 2018): Lernbare Task-Gewichte
  ``log_var_time`` und ``log_var_cost`` balancieren beide Tasks automatisch.
* **Consistency-Loss** (normierter Raum): Σ vorhergesagte Schritte ≈ Gesamtzeit/-kosten.
* **Noise-Injektion** (Phase 2+): Multiplikatives Rauschen auf Vorschritt-Features,
  mit optionalen per-Token-Overrides für Phase 3.
* **Zwei Trainingsphasen**:

  1. Phase 1 (Epochen 0 … ``freeze_encoder_epochs-1``): Encoder eingefroren.
  2. Phase 2 (ab Epoche ``freeze_encoder_epochs``): Encoder mit niedrigerer LR.

Verwendung
----------
::

    module = MTLStepTimeModule(lr=1e-4, embed_dim=128, ...)
    module = MTLStepTimeModule.from_pretrained_encoder(
        ckpt_path="path/to/step_time.ckpt",
        lr=1e-4, ...
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

from mpp.constants import VOCAB, INV_VOCAB
from mpp.ml.models.regressor.trsfm_encoder_regressor import TrsfmEncoderRegressor
from mpp.ml.models.regressor.mtl_step_decoder import MTLStepTimeDecoder, MTLEncoderDecoderModel

logger = logging.getLogger(__name__)


class MTLStepTimeModule(pl.LightningModule):
    """Lightning-Wrapper für MTL-Schritt-Zeit-und-Kosten-Vorhersage.

    Parameters
    ----------
    input_dim : int
        Dimensionalität der Eingabevektoren (Standard: 32).
    embed_dim : int
        Embedding-Dimension. Standard: 128.
    num_heads : int
        Anzahl Attention-Köpfe. Standard: 8.
    num_encoder_layers : int
        Anzahl Encoder-Layer. Standard: 4.
    num_decoder_layers : int
        Anzahl Decoder-Layer. Standard: 4.
    dropout : float
        Dropout-Rate. Standard: 0.1.
    max_seq_len : int
        Maximale Sequenzlänge. Standard: 12.
    lr : float
        Lernrate (Decoder + Kendall-Parameter). Standard: 1e-4.
    encoder_lr_factor : float
        Encoder-LR = lr × factor in Phase 2. Standard: 0.1.
    weight_decay : float
        Weight-Decay. Standard: 0.01.
    max_epochs : int
        Gesamtanzahl Epochen (für LR-Scheduler). Standard: 220.
    use_scheduler : bool
        CosineAnnealingLR verwenden? Standard: True.
    target_mean_time, target_std_time : float
        Normalisierungsparameter für Schrittzeiten.
    target_mean_cost, target_std_cost : float
        Normalisierungsparameter für Schrittkosten.
    lambda_consistency_time : float
        Gewicht λ_t für den Zeit-Consistency-Loss. Standard: 0.1.
    lambda_consistency_cost : float
        Gewicht λ_c für den Kosten-Consistency-Loss. Standard: 0.1.
    freeze_encoder_epochs : int
        Anzahl Epochen für Phase 1 (Encoder eingefroren). Standard: 20.
    noise_scale_time : float
        Multiplikativer Rausch-Anteil für Vorschritt-Zeiten in Phase 2.
        0.0 = kein Rauschen. Standard: 0.0.
    noise_scale_cost : float
        Multiplikativer Rausch-Anteil für Vorschritt-Kosten in Phase 2.
        0.0 = kein Rauschen. Standard: 0.0.
    noise_overrides_time : dict
        Token-Name → Noise-Skala (überschreibt ``noise_scale_time``).
        Beispiel: ``{"schweißen": 0.25, "bohren": 0.05}``.
    noise_overrides_cost : dict
        Token-Name → Noise-Skala (überschreibt ``noise_scale_cost``).
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
        max_epochs: int = 220,
        use_scheduler: bool = True,
        target_mean_time: float = 0.0,
        target_std_time: float = 1.0,
        target_mean_cost: float = 0.0,
        target_std_cost: float = 1.0,
        lambda_consistency_time: float = 0.1,
        lambda_consistency_cost: float = 0.1,
        freeze_encoder_epochs: int = 20,
        noise_scale_time: float = 0.0,
        noise_scale_cost: float = 0.0,
        noise_overrides_time: dict | None = None,
        noise_overrides_cost: dict | None = None,
        zero_cost_token_ids: list | None = None,
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
        decoder = MTLStepTimeDecoder(
            vocab_size=len(VOCAB),
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.model = MTLEncoderDecoderModel(encoder, decoder)

        # Lernbare Task-Gewichte (Kendall et al. 2018)
        self.log_var_time = nn.Parameter(torch.zeros(1))
        self.log_var_cost = nn.Parameter(torch.zeros(1))

        # Huber-Loss elementweise (PAD-Maskierung)
        self.huber = nn.HuberLoss(delta=1.0, reduction="none")

    # ------------------------------------------------------------------
    # Normalisierungs-Hilfsmethoden
    # ------------------------------------------------------------------

    def _normalize_time(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.hparams.target_mean_time) / self.hparams.target_std_time

    def _denormalize_time(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.hparams.target_std_time + self.hparams.target_mean_time

    def _normalize_cost(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.hparams.target_mean_cost) / self.hparams.target_std_cost

    def _denormalize_cost(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.hparams.target_std_cost + self.hparams.target_mean_cost

    # ------------------------------------------------------------------
    # Forward (Teacher Forcing)
    # ------------------------------------------------------------------

    def forward(
        self,
        vecset: torch.Tensor,
        step_tokens: torch.Tensor,
        prev_times: torch.Tensor,
        prev_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-Forcing-Forward (normalisierter Raum)."""
        return self.model(vecset, step_tokens, prev_times, prev_costs)

    @torch.no_grad()
    def generate(
        self,
        vecset: torch.Tensor,
        step_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive Inferenz – gibt absolute Zeiten und Kosten zurück.

        Für Tokens in ``zero_cost_token_ids`` (z. B. 'prüfen', 'kontrollieren')
        wird die Kostenvorhersage auf 0.0 gesetzt, da deren Kosten regelbasiert
        und nicht durch das Modell bestimmt werden.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(pred_times [B, seq_len], pred_costs [B, seq_len])`` in absoluten
            Einheiten (Minuten bzw. Dollar). PAD-Positionen sind 0.0.
        """
        pred_t_norm, pred_c_norm = self.model.generate(vecset, step_tokens)

        pred_times = self._denormalize_time(pred_t_norm)
        pred_costs = self._denormalize_cost(pred_c_norm)

        # Null-Kosten-Tokens auf 0 setzen – nach Denormalisierung,
        # da 0 im normierten Raum ≠ 0 in absoluten Einheiten.
        cost_ignore = self._cost_ignore_mask(step_tokens)
        pred_costs = pred_costs.masked_fill(cost_ignore, 0.0)

        return pred_times, pred_costs

    def generate_stream(
        self,
        vecset: torch.Tensor,
        step_tokens: torch.Tensor,
    ):
        """Autoregressive Inferenz als Generator – ein Ergebnis pro Schritt.

        Für Tokens in ``zero_cost_token_ids`` wird ``cost_dollar = 0.0``
        zurückgegeben; ``cost_is_rule_based = True`` zeigt dies an.

        Yields
        ------
        tuple[int, int, float, float, bool]
            ``(schritt_index, token_id, zeit_min, cost_dollar, cost_is_rule_based)``
        """
        zero_cost_ids = set()
        for name in (self.hparams.zero_cost_token_ids or []):
            if name in VOCAB:
                zero_cost_ids.add(VOCAB[name])

        for step_idx, token_id, t_abs, c_abs in self.model.generate_stream(
            vecset, step_tokens,
            target_mean_time=self.hparams.target_mean_time,
            target_std_time=self.hparams.target_std_time,
            target_mean_cost=self.hparams.target_mean_cost,
            target_std_cost=self.hparams.target_std_cost,
        ):
            is_rule_based = token_id in zero_cost_ids
            yield step_idx, token_id, t_abs, 0.0 if is_rule_based else c_abs, is_rule_based

    # ------------------------------------------------------------------
    # Hilfsmethode: Kosten-Ignorier-Maske
    # ------------------------------------------------------------------

    def _cost_ignore_mask(self, step_tokens: torch.Tensor) -> torch.Tensor:
        """PAD-Maske erweitert um Tokens mit definitionsgemäß Kosten = 0.

        Tokens wie 'prüfen' und 'kontrollieren' haben im Datensatz immer
        Kosten = 0 (regelbasiert berechnet), daher werden sie aus dem
        Cost-Huber-Loss und den Cost-Konsistenz- und Metrik-Berechnungen
        ausgeschlossen.
        """
        mask = step_tokens == VOCAB["PAD"]
        for name in (self.hparams.zero_cost_token_ids or []):
            if name in VOCAB:
                mask = mask | (step_tokens == VOCAB[name])
        return mask

    # ------------------------------------------------------------------
    # Noise-Injektion (Phase 2+)
    # ------------------------------------------------------------------

    def _token_noise_tensor(self, step_tokens: torch.Tensor, task: str) -> torch.Tensor:
        """Erstellt einen [B, S]-Tensor mit der pro-Token-Noise-Skala."""
        if task == "time":
            default_scale = self.hparams.noise_scale_time
            overrides = self.hparams.noise_overrides_time or {}
        else:
            default_scale = self.hparams.noise_scale_cost
            overrides = self.hparams.noise_overrides_cost or {}

        noise = torch.full_like(step_tokens, default_scale, dtype=torch.float32)
        for token_name, scale in overrides.items():
            if token_name in VOCAB:
                noise[step_tokens == VOCAB[token_name]] = scale
        return noise

    def _apply_noise(
        self,
        step_tokens: torch.Tensor,
        step_times_norm: torch.Tensor,
        step_costs_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Noise-Injektion in Phase 2+; in Phase 1 kein Rauschen."""
        if self.current_epoch < self.hparams.freeze_encoder_epochs:
            return step_times_norm, step_costs_norm

        noise_t = self._token_noise_tensor(step_tokens, "time")   # [B, S]
        noise_c = self._token_noise_tensor(step_tokens, "cost")   # [B, S]

        noisy_t = step_times_norm * (1.0 + torch.randn_like(step_times_norm) * noise_t)
        noisy_c = step_costs_norm * (1.0 + torch.randn_like(step_costs_norm) * noise_c)
        return noisy_t, noisy_c

    # ------------------------------------------------------------------
    # Loss-Berechnung (normierter Raum durchgehend)
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        pred_t_norm: torch.Tensor,
        pred_c_norm: torch.Tensor,
        gt_t_norm: torch.Tensor,
        gt_c_norm: torch.Tensor,
        step_tokens: torch.Tensor,
        total_time: torch.Tensor,
        total_cost: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Kendall-MTL-Loss + Consistency-Loss (alle im normierten Raum).

        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor]
            ``(gesamt_loss, huber_time, huber_cost, consistency_loss)``
        """
        pad_mask        = step_tokens == VOCAB["PAD"]                   # [B, seq_len]
        cost_ignore     = self._cost_ignore_mask(step_tokens)            # PAD + Null-Kosten-Tokens

        valid_count      = (~pad_mask).float().sum().clamp(min=1)        # Skalar
        cost_valid_count = (~cost_ignore).float().sum().clamp(min=1)     # Skalar

        n_valid_time = (~pad_mask).float().sum(dim=-1).clamp(min=1)      # [B]
        n_valid_cost = (~cost_ignore).float().sum(dim=-1).clamp(min=1)   # [B]

        # --- Huber pro Schritt (normiert, maskiert) ---
        huber_t_elem = self.huber(pred_t_norm, gt_t_norm)                # [B, seq_len]
        huber_c_elem = self.huber(pred_c_norm, gt_c_norm)
        huber_t = huber_t_elem.masked_fill(pad_mask,    0.0).sum() / valid_count
        huber_c = huber_c_elem.masked_fill(cost_ignore, 0.0).sum() / cost_valid_count

        # --- Kendall MTL ---
        # L = exp(-s) * Huber + s    (s = log_var, zur Optimierung stabiler)
        mtl_loss = (
            torch.exp(-self.log_var_time) * huber_t + self.log_var_time
            + torch.exp(-self.log_var_cost) * huber_c + self.log_var_cost
        )

        # --- Consistency-Loss im normierten Raum ---
        # Normierung: sum(norm_i) = (total - n_valid * mean) / std
        # Für Kosten: n_valid_cost zählt nur Tokens mit echten Kosten,
        # da total_cost keine Beiträge von Null-Kosten-Tokens enthält.
        total_t_norm_sum = (
            (total_time - n_valid_time * self.hparams.target_mean_time)
            / self.hparams.target_std_time
        )
        total_c_norm_sum = (
            (total_cost - n_valid_cost * self.hparams.target_mean_cost)
            / self.hparams.target_std_cost
        )
        pred_t_sum = pred_t_norm.masked_fill(pad_mask,    0.0).sum(dim=-1)  # [B]
        pred_c_sum = pred_c_norm.masked_fill(cost_ignore, 0.0).sum(dim=-1)  # [B]

        consistency = (
            self.hparams.lambda_consistency_time * F.mse_loss(pred_t_sum, total_t_norm_sum)
            + self.hparams.lambda_consistency_cost * F.mse_loss(pred_c_sum, total_c_norm_sum)
        )

        total_loss = mtl_loss + consistency
        return total_loss, huber_t, huber_c, consistency

    # ------------------------------------------------------------------
    # Lightning-Steps
    # ------------------------------------------------------------------

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> torch.Tensor:
        vecset, step_tokens, step_times, step_costs, total_time, total_cost = batch

        step_times_norm = self._normalize_time(step_times)
        step_costs_norm = self._normalize_cost(step_costs)

        # Noise auf Vorschritt-Features anwenden (Phase 2+)
        noisy_times_norm, noisy_costs_norm = self._apply_noise(
            step_tokens, step_times_norm, step_costs_norm
        )

        # Teacher Forcing: prev = GT, nach rechts verschoben (mit Noise)
        prev_times = F.pad(noisy_times_norm[:, :-1], (1, 0), value=0.0)
        prev_costs = F.pad(noisy_costs_norm[:, :-1], (1, 0), value=0.0)

        pred_t_norm, pred_c_norm = self.model(vecset, step_tokens, prev_times, prev_costs)

        loss, huber_t, huber_c, consistency = self._compute_loss(
            pred_t_norm, pred_c_norm,
            step_times_norm, step_costs_norm,
            step_tokens, total_time, total_cost,
        )

        self.log("train_loss",        loss,        on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_huber_time",  huber_t,     on_step=False, on_epoch=True)
        self.log("train_huber_cost",  huber_c,     on_step=False, on_epoch=True)
        self.log("train_consistency", consistency, on_step=False, on_epoch=True)
        self.log("log_var_time",      self.log_var_time.item(), on_step=False, on_epoch=True)
        self.log("log_var_cost",      self.log_var_cost.item(), on_step=False, on_epoch=True)

        opt = self.optimizers()
        self.log("lr_encoder", opt.param_groups[0]["lr"], on_step=False, on_epoch=True)
        self.log("lr_decoder", opt.param_groups[1]["lr"], on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> None:
        vecset, step_tokens, step_times, step_costs, total_time, total_cost = batch

        step_times_norm = self._normalize_time(step_times)
        step_costs_norm = self._normalize_cost(step_costs)

        # Validation immer ohne Noise, Teacher Forcing
        prev_times = F.pad(step_times_norm[:, :-1], (1, 0), value=0.0)
        prev_costs = F.pad(step_costs_norm[:, :-1], (1, 0), value=0.0)

        pred_t_norm, pred_c_norm = self.model(vecset, step_tokens, prev_times, prev_costs)

        loss, huber_t, huber_c, consistency = self._compute_loss(
            pred_t_norm, pred_c_norm,
            step_times_norm, step_costs_norm,
            step_tokens, total_time, total_cost,
        )

        # Metriken in absoluten Einheiten (Null-Kosten-Tokens aus Cost-Metriken ausschließen)
        pad_mask    = step_tokens == VOCAB["PAD"]
        cost_ignore = self._cost_ignore_mask(step_tokens)

        pred_t_abs = self._denormalize_time(pred_t_norm).masked_fill(pad_mask,    0.0)
        pred_c_abs = self._denormalize_cost(pred_c_norm).masked_fill(cost_ignore, 0.0)
        gt_t_abs   = step_times.masked_fill(pad_mask,    0.0)
        gt_c_abs   = step_costs.masked_fill(cost_ignore, 0.0)

        valid_t = (~pad_mask).float().sum().clamp(min=1)
        valid_c = (~cost_ignore).float().sum().clamp(min=1)
        mae_t  = (pred_t_abs - gt_t_abs).abs().sum() / valid_t
        mae_c  = (pred_c_abs - gt_c_abs).abs().sum() / valid_c
        rmse_t = ((pred_t_abs - gt_t_abs) ** 2).sum().div(valid_t).sqrt()
        rmse_c = ((pred_c_abs - gt_c_abs) ** 2).sum().div(valid_c).sqrt()

        self.log("val_loss",         loss,        on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_huber_time",   huber_t,     on_step=False, on_epoch=True)
        self.log("val_huber_cost",   huber_c,     on_step=False, on_epoch=True)
        self.log("val_consistency",  consistency, on_step=False, on_epoch=True)
        self.log("val_mae_time",     mae_t,       on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae_cost",     mae_c,       on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse_time",    rmse_t,      on_step=False, on_epoch=True)
        self.log("val_rmse_cost",    rmse_c,      on_step=False, on_epoch=True)
        self.log("log_var_time",     self.log_var_time.item(), on_step=False, on_epoch=True)
        self.log("log_var_cost",     self.log_var_cost.item(), on_step=False, on_epoch=True)

    def test_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int
    ) -> None:
        vecset, step_tokens, step_times, step_costs, total_time, total_cost = batch

        step_times_norm = self._normalize_time(step_times)
        step_costs_norm = self._normalize_cost(step_costs)
        prev_times = F.pad(step_times_norm[:, :-1], (1, 0), value=0.0)
        prev_costs = F.pad(step_costs_norm[:, :-1], (1, 0), value=0.0)

        pred_t_norm, pred_c_norm = self.model(vecset, step_tokens, prev_times, prev_costs)

        loss, huber_t, huber_c, consistency = self._compute_loss(
            pred_t_norm, pred_c_norm,
            step_times_norm, step_costs_norm,
            step_tokens, total_time, total_cost,
        )

        pad_mask    = step_tokens == VOCAB["PAD"]
        cost_ignore = self._cost_ignore_mask(step_tokens)

        pred_t_abs = self._denormalize_time(pred_t_norm).masked_fill(pad_mask,    0.0)
        pred_c_abs = self._denormalize_cost(pred_c_norm).masked_fill(cost_ignore, 0.0)
        gt_t_abs   = step_times.masked_fill(pad_mask,    0.0)
        gt_c_abs   = step_costs.masked_fill(cost_ignore, 0.0)

        valid_t = (~pad_mask).float().sum().clamp(min=1)
        valid_c = (~cost_ignore).float().sum().clamp(min=1)
        mae_t  = (pred_t_abs - gt_t_abs).abs().sum() / valid_t
        mae_c  = (pred_c_abs - gt_c_abs).abs().sum() / valid_c
        rmse_t = ((pred_t_abs - gt_t_abs) ** 2).sum().div(valid_t).sqrt()
        rmse_c = ((pred_c_abs - gt_c_abs) ** 2).sum().div(valid_c).sqrt()

        self.log("test_loss",        loss,        on_step=False, on_epoch=True)
        self.log("test_huber_time",  huber_t,     on_step=False, on_epoch=True)
        self.log("test_huber_cost",  huber_c,     on_step=False, on_epoch=True)
        self.log("test_consistency", consistency, on_step=False, on_epoch=True)
        self.log("test_mae_time",    mae_t,       on_step=False, on_epoch=True)
        self.log("test_mae_cost",    mae_c,       on_step=False, on_epoch=True)
        self.log("test_rmse_time",   rmse_t,      on_step=False, on_epoch=True)
        self.log("test_rmse_cost",   rmse_c,      on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Encoder-Freeze-Logik
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        """Phase 1 → Phase 2: Encoder ein-/auftauen."""
        freeze = self.hparams.freeze_encoder_epochs
        encoder_params = list(self.model.encoder.parameters())

        if self.current_epoch < freeze:
            for p in encoder_params:
                p.requires_grad = False
        elif self.current_epoch == freeze:
            logger.info(
                f"Epoche {self.current_epoch}: Phase 2 – Encoder aufgetaut "
                f"(LR = {self.hparams.lr * self.hparams.encoder_lr_factor:.2e})."
            )
            for p in encoder_params:
                p.requires_grad = True

    # ------------------------------------------------------------------
    # MLflow-Logging beim Trainingsstart
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
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
            "flash_attention_active":    flash_active,
            "training_device":           str(device),
            "training_precision":        str(precision),
            "batch_size":                self.trainer.train_dataloader.batch_size,
            "target_mean_time":          self.hparams.target_mean_time,
            "target_std_time":           self.hparams.target_std_time,
            "target_mean_cost":          self.hparams.target_mean_cost,
            "target_std_cost":           self.hparams.target_std_cost,
            "freeze_encoder_epochs":     self.hparams.freeze_encoder_epochs,
            "lambda_consistency_time":   self.hparams.lambda_consistency_time,
            "lambda_consistency_cost":   self.hparams.lambda_consistency_cost,
        })

    # ------------------------------------------------------------------
    # Optimizer: zwei Parametergruppen + optionaler LR-Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        """Encoder mit niedrigerer LR, Decoder + Kendall-Parameter mit voller LR."""
        encoder_params = list(self.model.encoder.parameters())
        other_params   = (
            list(self.model.decoder.parameters())
            + [self.log_var_time, self.log_var_cost]
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.hparams.lr * self.hparams.encoder_lr_factor},
                {"params": other_params,   "lr": self.hparams.lr},
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
    # Klassenmethode: vortrainierter Encoder aus StepTimeRegressionModule
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained_encoder(
        cls,
        ckpt_path: str,
        **kwargs: Any,
    ) -> "MTLStepTimeModule":
        """Erstellt ein neues Modul und lädt Encoder-Gewichte aus einem
        ``StepTimeRegressionModule``-Checkpoint.

        Nur ``embedding``- und ``encoder``-Gewichte werden übernommen;
        Decoder und Kendall-Parameter werden neu initialisiert.

        NOTE: ``embed_dim``, ``num_heads`` und ``num_encoder_layers`` in
        ``kwargs`` müssen mit dem geladenen Checkpoint übereinstimmen.

        Parameters
        ----------
        ckpt_path : str
            Pfad zur ``*.ckpt``-Datei.
        **kwargs
            Initialisierungsargumente für :class:`MTLStepTimeModule`.

        Returns
        -------
        MTLStepTimeModule
        """
        from mpp.ml.models.regressor.step_time_regression_module import StepTimeRegressionModule

        pretrained = StepTimeRegressionModule.load_from_checkpoint(ckpt_path)
        pretrained_encoder = pretrained.model.encoder

        # Encoder-Architektur immer aus dem Checkpoint übernehmen,
        # damit Abweichungen in der Config keinen size-mismatch verursachen.
        ckpt_embed_dim         = pretrained.hparams.embed_dim
        ckpt_num_encoder_layers = pretrained.hparams.num_encoder_layers
        ckpt_num_heads         = pretrained.hparams.num_heads

        if kwargs.get("embed_dim", ckpt_embed_dim) != ckpt_embed_dim:
            logger.warning(
                f"embed_dim in kwargs ({kwargs['embed_dim']}) weicht vom Checkpoint "
                f"({ckpt_embed_dim}) ab – verwende Checkpoint-Wert."
            )
        if kwargs.get("num_encoder_layers", ckpt_num_encoder_layers) != ckpt_num_encoder_layers:
            logger.warning(
                f"num_encoder_layers in kwargs ({kwargs['num_encoder_layers']}) weicht vom "
                f"Checkpoint ({ckpt_num_encoder_layers}) ab – verwende Checkpoint-Wert."
            )

        kwargs["embed_dim"]          = ckpt_embed_dim
        kwargs["num_encoder_layers"] = ckpt_num_encoder_layers
        kwargs["num_heads"]          = ckpt_num_heads

        module = cls(**kwargs)
        module.model.encoder.embedding.load_state_dict(
            pretrained_encoder.embedding.state_dict()
        )
        module.model.encoder.encoder.load_state_dict(
            pretrained_encoder.encoder.state_dict()
        )
        logger.info(
            f"Encoder-Gewichte aus '{ckpt_path}' geladen "
            f"(embed_dim={ckpt_embed_dim}, num_encoder_layers={ckpt_num_encoder_layers}, "
            f"num_heads={ckpt_num_heads})."
        )
        return module
