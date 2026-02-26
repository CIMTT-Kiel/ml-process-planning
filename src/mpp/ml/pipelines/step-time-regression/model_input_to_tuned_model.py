"""Pipeline: Schrittweise Prozesszeit-Regression (step-time-regression)

Ablauf
------
1. Normalisierungsstatistiken der per-Schritt-Zeiten aus Trainingsdaten berechnen.
2. Optionaler Encoder-Warmstart aus vortrainiertem ProcessRegressionModule-Checkpoint.
3. Hyperparameter-Tuning mit Optuna (Konfiguration in
   ``config/step_time_regression.yaml``).
4. Finales Training mit den besten Hyperparametern und zwei Phasen:
   - Phase 1 (0 … freeze_encoder_epochs-1): Nur Decoder trainiert.
   - Phase 2 (ab freeze_encoder_epochs):    Encoder mit niedrigerer LR.

Konfiguration
-------------
Alle Parameter werden aus ``config/step_time_regression.yaml``
(+ ``config/base.yaml``) gelesen.

Ausführung
----------
::

    python -m "mpp.ml.pipelines.step-time-regression.model_input_to_tuned_model"
"""

from __future__ import annotations

import logging

import mlflow
import torch

from mpp.constants import PATHS, VOCAB
from mpp.ml.callbacks.artifact_callbacks import (
    BestStepTimeModelPlotCallback,
    StepTimePredictionPlotCallback,
)
from mpp.ml.models.regressor.step_time_regression_module import StepTimeRegressionModule
from mpp.ml.pipelines.base_pipeline import (
    build_callbacks,
    build_mlflow_logger,
    build_trainer,
    get_dataloaders,
    load_config,
    run_tuning,
    suggest_hyperparams,
)

logger = logging.getLogger(__name__)

_CFG_PATH = PATHS.CONFIG / "step_time_regression.yaml"
cfg = load_config(_CFG_PATH)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def compute_step_time_stats(train_loader) -> tuple[float, float]:
    """Berechnet Mean und Std der *gefilterten* per-Schritt-Zeiten.

    Nur valide (nicht-PAD) Positionen gehen in die Statistik ein, damit
    das Padding die Normalisierung nicht verzerrt.

    Parameters
    ----------
    train_loader : DataLoader
        Trainings-DataLoader mit ``target_type="step-time"``
        (Batches: ``vecset, step_tokens, step_times, total_time``).

    Returns
    -------
    tuple[float, float]
        ``(mean, std)`` der per-Schritt-Zeiten in Minuten.
    """
    all_times: list[torch.Tensor] = []
    for _, step_tokens, step_times, _ in train_loader:
        valid_mask = step_tokens != VOCAB["PAD"]               # [B, seq_len]
        all_times.append(step_times[valid_mask])               # flattened valid values
    all_times_t = torch.cat(all_times)
    mean = all_times_t.mean().item()
    std  = all_times_t.std().item()
    logger.info(f"Schrittzeiten: mean={mean:.2f} min, std={std:.2f} min")
    return mean, std


def _build_model(hp: dict, mean: float, std: float, max_epochs: int) -> StepTimeRegressionModule:
    """Erstellt ein StepTimeRegressionModule aus gesampelten Hyperparametern.

    Wenn in der Config ``training.pretrained_encoder_ckpt`` gesetzt ist,
    werden die Encoder-Gewichte aus dem angegebenen Checkpoint geladen.
    """
    common_kwargs: dict = dict(
        lr=hp["lr"],
        embed_dim=hp["embed_dim"],
        num_encoder_layers=hp["num_encoder_layers"],
        num_decoder_layers=hp["num_decoder_layers"],
        dropout=hp["dropout"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=max_epochs,
        target_mean=mean,
        target_std=std,
        lambda_consistency=cfg["training"]["lambda_consistency"],
        freeze_encoder_epochs=cfg["training"]["freeze_encoder_epochs"],
        encoder_lr_factor=cfg["training"]["encoder_lr_factor"],
        scheduled_sampling=cfg["training"]["scheduled_sampling"],
        scheduled_sampling_rate=cfg["training"]["scheduled_sampling_rate"],
    )

    ckpt = cfg["training"].get("pretrained_encoder_ckpt")
    if ckpt:
        logger.info(f"Lade vortrainierte Encoder-Gewichte aus: {ckpt}")
        return StepTimeRegressionModule.from_pretrained_encoder(
            ckpt_path=ckpt,
            use_scheduler=False,
            **common_kwargs,
        )
    return StepTimeRegressionModule(use_scheduler=False, **common_kwargs)


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main() -> None:
    train_loader, val_loader = get_dataloaders(cfg)

    # Normalisierungsparameter einmalig aus Trainingsdaten berechnen
    target_mean, target_std = compute_step_time_stats(train_loader)

    # ------------------------------------------------------------------
    # Hyperparameter-Tuning
    # ------------------------------------------------------------------
    def objective(trial):
        hp = suggest_hyperparams(trial, cfg["hyperparameter_search"])
        max_epochs = cfg["training"]["tuning_epochs"]
        torch.set_float32_matmul_precision("medium")

        model = _build_model(hp, target_mean, target_std, max_epochs)

        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True) as child_run:
            mlf_logger = build_mlflow_logger(
                cfg,
                cfg["mlflow"]["tuning_experiment_name"],
                run_id=child_run.info.run_id,
            )
            callbacks = build_callbacks(
                cfg,
                cfg["checkpoint"]["tuning_subdir"],
                cfg["checkpoint"]["filename"],
                patience=cfg["training"]["tuning_patience"],
            )
            callbacks.append(StepTimePredictionPlotCallback(
                plot_every_n_epochs=cfg["training"]["plot_every_n_epochs"],
            ))
            callbacks.append(BestStepTimeModelPlotCallback())
            trainer = build_trainer(cfg, max_epochs, mlf_logger, callbacks)
            trainer.fit(model, train_loader, val_loader)

            val_loss = trainer.callback_metrics["val_loss"].item()
            mlf_logger.log_hyperparams(trial.params)
            mlf_logger.log_metrics({"val_loss": val_loss})

        return val_loss

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["tuning_experiment_name"])
    with mlflow.start_run(run_name="optuna-study"):
        study = run_tuning(cfg, objective)

    # ------------------------------------------------------------------
    # Finales Training mit besten Hyperparametern
    # ------------------------------------------------------------------
    best = study.best_trial.params
    torch.set_float32_matmul_precision("high")

    final_common_kwargs: dict = dict(
        lr=best["lr"],
        embed_dim=best["embed_dim"],
        num_encoder_layers=best["num_encoder_layers"],
        num_decoder_layers=best["num_decoder_layers"],
        dropout=best["dropout"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=cfg["training"]["final_epochs"],
        target_mean=target_mean,
        target_std=target_std,
        lambda_consistency=cfg["training"]["lambda_consistency"],
        freeze_encoder_epochs=cfg["training"]["freeze_encoder_epochs"],
        encoder_lr_factor=cfg["training"]["encoder_lr_factor"],
        scheduled_sampling=cfg["training"]["scheduled_sampling"],
        scheduled_sampling_rate=cfg["training"]["scheduled_sampling_rate"],
    )

    ckpt = cfg["training"].get("pretrained_encoder_ckpt")
    if ckpt:
        logger.info(f"Finales Training: Lade vortrainierte Encoder-Gewichte aus: {ckpt}")
        final_model = StepTimeRegressionModule.from_pretrained_encoder(
            ckpt_path=ckpt,
            **final_common_kwargs,
        )
    else:
        final_model = StepTimeRegressionModule(**final_common_kwargs)

    mlf_logger = build_mlflow_logger(
        cfg, cfg["mlflow"]["experiment_name"], run_name="best-model"
    )
    callbacks = build_callbacks(
        cfg,
        cfg["checkpoint"]["best_subdir"],
        cfg["checkpoint"]["filename"],
        patience=cfg["training"]["final_patience"],
    )
    callbacks.append(StepTimePredictionPlotCallback(
        plot_every_n_epochs=cfg["training"]["plot_every_n_epochs"],
    ))
    callbacks.append(BestStepTimeModelPlotCallback())
    trainer = build_trainer(
        cfg, cfg["training"]["final_epochs"], mlf_logger, callbacks
    )
    trainer.fit(final_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
