"""Pipeline: MTL-Schritt-Zeit-und-Kosten-Regression (time-and-cost-per-step)

Ablauf
------
1. Normalisierungsstatistiken für Zeit und Kosten aus Trainingsdaten berechnen.
2. Optionaler Encoder-Warmstart aus vortrainiertem StepTimeRegressionModule-Checkpoint.
3. Hyperparameter-Tuning mit Optuna (Konfiguration in
   ``config/time_and_cost_regression.yaml``).
4. Finales Training mit den besten Hyperparametern und zwei Phasen:
   - Phase 1 (0 … freeze_encoder_epochs-1): Encoder eingefroren, nur Decoder + Kendall-Gewichte.
   - Phase 2 (ab freeze_encoder_epochs):     Encoder mit niedrigerer LR + Noise-Injektion.

Konfiguration
-------------
Alle Parameter aus ``config/time_and_cost_regression.yaml`` (+ ``config/base.yaml``).

Ausführung
----------
::

    python -m "mpp.ml.pipelines.time-and-cost-per-step.model_input_to_tuned_model"
"""

from __future__ import annotations

import logging

import mlflow
import torch
from pytorch_lightning import Callback

from mpp.constants import PATHS, VOCAB
from mpp.ml.callbacks.artifact_callbacks import (
    BestMTLModelPlotCallback,
    MTLPredictionPlotCallback,
)
from mpp.ml.models.regressor.mtl_step_regression_module import MTLStepTimeModule
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

_CFG_PATH = PATHS.CONFIG / "time_and_cost_regression.yaml"
cfg = load_config(_CFG_PATH)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def compute_mtl_stats(train_loader) -> tuple[float, float, float, float]:
    """Berechnet Mean und Std der per-Schritt-Zeiten und -Kosten.

    Nur valide (nicht-PAD) Positionen gehen in die Statistik ein.

    Parameters
    ----------
    train_loader : DataLoader
        Trainings-DataLoader mit ``target_type="step-time-cost"``.
        Batches: ``(vecset, step_tokens, step_times, step_costs, total_time, total_cost)``.

    Returns
    -------
    tuple[float, float, float, float]
        ``(mean_time, std_time, mean_cost, std_cost)``
    """
    all_times: list[torch.Tensor] = []
    all_costs: list[torch.Tensor] = []
    for _, step_tokens, step_times, step_costs, _, _ in train_loader:
        mask = step_tokens != VOCAB["PAD"]
        all_times.append(step_times[mask])
        all_costs.append(step_costs[mask])

    times_t = torch.cat(all_times)
    costs_t = torch.cat(all_costs)

    mean_t = times_t.mean().item()
    std_t  = times_t.std().item()
    mean_c = costs_t.mean().item()
    std_c  = costs_t.std().item()

    logger.info(f"Schrittzeiten:  mean={mean_t:.2f} min, std={std_t:.2f} min")
    logger.info(f"Schrittkosten: mean={mean_c:.2f} $,   std={std_c:.2f} $")
    return mean_t, std_t, mean_c, std_c


def _build_model(
    hp: dict,
    stats: tuple[float, float, float, float],
    max_epochs: int,
    use_scheduler: bool = False,
) -> MTLStepTimeModule:
    """Erstellt ein MTLStepTimeModule aus gesampelten Hyperparametern.

    Wenn in der Config ``training.pretrained_encoder_ckpt`` gesetzt ist,
    werden die Encoder-Gewichte aus dem angegebenen Checkpoint geladen.
    """
    mean_t, std_t, mean_c, std_c = stats
    tr = cfg["training"]

    common_kwargs: dict = dict(
        lr=hp["lr"],
        embed_dim=tr["embed_dim"],            # fest: muss mit Encoder-Checkpoint übereinstimmen
        num_encoder_layers=tr["num_encoder_layers"],  # fest: idem
        num_decoder_layers=hp["num_decoder_layers"],
        dropout=hp["dropout"],
        weight_decay=tr["weight_decay"],
        max_epochs=max_epochs,
        use_scheduler=use_scheduler,
        target_mean_time=mean_t,
        target_std_time=std_t,
        target_mean_cost=mean_c,
        target_std_cost=std_c,
        lambda_consistency_time=tr["lambda_consistency_time"],
        lambda_consistency_cost=tr["lambda_consistency_cost"],
        freeze_encoder_epochs=tr["freeze_encoder_epochs"],
        encoder_lr_factor=tr["encoder_lr_factor"],
        noise_scale_time=tr.get("noise_scale_time", 0.0),
        noise_scale_cost=tr.get("noise_scale_cost", 0.0),
        noise_overrides_time=tr.get("noise_overrides_time", {}),
        noise_overrides_cost=tr.get("noise_overrides_cost", {}),
    )

    ckpt = tr.get("pretrained_encoder_ckpt")
    if ckpt:
        logger.info(f"Lade vortrainierte Encoder-Gewichte aus: {ckpt}")
        return MTLStepTimeModule.from_pretrained_encoder(ckpt_path=ckpt, **common_kwargs)
    return MTLStepTimeModule(**common_kwargs)


# ---------------------------------------------------------------------------
# Phase-1-Evaluation-Callback
# ---------------------------------------------------------------------------

class Phase1EvaluationCallback(Callback):
    """Gibt am Ende von Phase 1 einen Noise-Kalibrierungshinweis aus.

    Am Ende der letzten Phase-1-Epoche (``freeze_encoder_epochs - 1``) werden
    die val-Metriken geloggt, damit der Anwender per-Token-Noise-Skalen für
    Phase 2 ableiten kann.
    """

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if trainer.current_epoch != pl_module.hparams.freeze_encoder_epochs - 1:
            return

        logger.info(
            "=== Ende Phase 1 – Noise-Kalibrierung empfohlen ==="
        )
        logger.info(
            "Prüfe val_mae_time / val_mae_cost aus MLflow, um noise_overrides "
            "pro Schritttyp in time_and_cost_regression.yaml zu setzen."
        )
        report = (
            "Phase 1 abgeschlossen.\n"
            "Empfehlung: Setze noise_overrides_time/<token> = val_mae_time_<token> / mean_time\n"
            "            Setze noise_overrides_cost/<token> = val_mae_cost_<token> / mean_cost\n"
        )
        try:
            mlflow.log_text(report, "noise_calibration_hint.txt")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------------------------

def main() -> None:
    train_loader, val_loader = get_dataloaders(cfg)

    # Normalisierungsparameter einmalig aus Trainingsdaten berechnen
    stats = compute_mtl_stats(train_loader)

    tr = cfg["training"]

    # ------------------------------------------------------------------
    # Hyperparameter-Tuning
    # ------------------------------------------------------------------
    def objective(trial):
        hp = suggest_hyperparams(trial, cfg["hyperparameter_search"])
        max_epochs = tr["tuning_epochs"]
        torch.set_float32_matmul_precision("medium")

        model = _build_model(hp, stats, max_epochs)

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
                patience=tr["tuning_patience"],
            )
            callbacks.append(MTLPredictionPlotCallback(
                plot_every_n_epochs=tr.get("plot_every_n_epochs", 5),
            ))
            callbacks.append(BestMTLModelPlotCallback())
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

    max_epochs_final = tr["freeze_encoder_epochs"] + tr["finetune_epochs"]
    final_model = _build_model(best, stats, max_epochs_final, use_scheduler=True)

    mlf_logger = build_mlflow_logger(
        cfg, cfg["mlflow"]["experiment_name"], run_name="best-model"
    )
    callbacks = build_callbacks(
        cfg,
        cfg["checkpoint"]["best_subdir"],
        cfg["checkpoint"]["filename"],
        patience=tr["final_patience"],
    )
    callbacks.extend([
        Phase1EvaluationCallback(),
        MTLPredictionPlotCallback(plot_every_n_epochs=tr.get("plot_every_n_epochs", 5)),
        BestMTLModelPlotCallback(),
    ])

    trainer = build_trainer(cfg, max_epochs_final, mlf_logger, callbacks)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    trainer.fit(final_model, train_loader, val_loader)


if __name__ == "__main__":
    main()
