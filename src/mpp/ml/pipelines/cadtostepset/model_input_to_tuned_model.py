"""
Pipeline: Prozessschritt-Klassifikation (cadtostepset)

Ablauf
------
1. Hyperparameter-Tuning mit Optuna (Anzahl Trials in config/cadtostepset.yaml)
2. Finales Training mit den besten Hyperparametern

Konfiguration
-------------
Alle Parameter werden aus config/cadtostepset.yaml (+ config/base.yaml) gelesen.
Zum Anpassen einfach die YAML-Dateien bearbeiten – kein Code-Änderung nötig.

Ausführung
----------
    python -m mpp.ml.pipelines.cadtostepset.model_input_to_tuned_model
"""

import logging

import mlflow
import torch

from mpp.constants import PATHS
from mpp.ml.callbacks.artifact_callbacks import (
    BestStepsetModelPlotCallback,
    StepsetPredictionPlotCallback,
)
from mpp.ml.models.classifier.cadtostepset import ProcessClassificationTrsfmEncoderModule
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

_CFG_PATH = PATHS.CONFIG / "cadtostepset.yaml"
cfg = load_config(_CFG_PATH)


def compute_pos_weight(train_loader, cap: float | None = None) -> torch.Tensor:
    """Berechnet pos_weight = neg_count / pos_count pro Klasse aus dem Trainings-Loader.

    Wird als Gewichtung für BCEWithLogitsLoss verwendet, um Klassen-Imbalance
    auszugleichen (z.B. schleifen/kontrollieren mit nur ~8% Häufigkeit → raw ~11.5).

    Parameters
    ----------
    cap : float or None
        Optionale Obergrenze für pos_weight. Verhindert, dass sehr seltene Klassen
        den Loss dominieren und Recall auf ~90% fixieren, während Precision leidet.
        Empfehlung: 4.0 (konfigurierbar in cadtostepset.yaml → training.pos_weight_cap).

    Returns
    -------
    torch.Tensor
        1D-Tensor der Form [num_classes] mit Gewichten ≥ 1 für seltene Klassen.
    """
    total = 0
    pos_counts = None
    for _, y in train_loader:
        pos_counts = y.sum(dim=0) if pos_counts is None else pos_counts + y.sum(dim=0)
        total += y.size(0)
    neg_counts = total - pos_counts
    pos_weight = (neg_counts / pos_counts.clamp(min=1)).float()
    if cap is not None:
        pos_weight = pos_weight.clamp(max=cap)
    logger.info(f"pos_weight berechnet (cap={cap}): {pos_weight.tolist()}")
    return pos_weight


def main():
    train_loader, val_loader = get_dataloaders(cfg)

    # pos_weight einmalig aus Trainingsdaten berechnen
    pos_weight_cap = cfg["training"].get("pos_weight_cap", None)
    pos_weight = compute_pos_weight(train_loader, cap=pos_weight_cap)

    # ------------------------------------------------------------------
    # Hyperparameter-Tuning
    # ------------------------------------------------------------------
    def objective(trial):
        hp = suggest_hyperparams(trial, cfg["hyperparameter_search"])
        max_epochs = cfg["training"]["tuning_epochs"]
        torch.set_float32_matmul_precision("medium")

        model = ProcessClassificationTrsfmEncoderModule(
            lr=hp["lr"],
            embed_dim=hp["embed_dim"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
            weight_decay=cfg["training"]["weight_decay"],
            max_epochs=max_epochs,
            use_scheduler=False,
            pos_weight=pos_weight,
        )

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
            callbacks.append(StepsetPredictionPlotCallback(
                plot_every_n_epochs=cfg["training"]["plot_every_n_epochs"],
            ))
            callbacks.append(BestStepsetModelPlotCallback())
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

    model = ProcessClassificationTrsfmEncoderModule(
        lr=best["lr"],
        embed_dim=best["embed_dim"],
        num_layers=best["num_layers"],
        dropout=best["dropout"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=cfg["training"]["final_epochs"],
        pos_weight=pos_weight,
    )

    mlf_logger = build_mlflow_logger(
        cfg, cfg["mlflow"]["experiment_name"], run_name="best-model"
    )
    callbacks = build_callbacks(
        cfg,
        cfg["checkpoint"]["best_subdir"],
        cfg["checkpoint"]["filename"],
        patience=cfg["training"]["final_patience"],
    )
    callbacks.append(StepsetPredictionPlotCallback(
        plot_every_n_epochs=cfg["training"]["plot_every_n_epochs"],
    ))
    callbacks.append(BestStepsetModelPlotCallback())
    trainer = build_trainer(cfg, cfg["training"]["final_epochs"], mlf_logger, callbacks)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
