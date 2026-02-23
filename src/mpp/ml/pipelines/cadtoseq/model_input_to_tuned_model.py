"""
Pipeline: Sequenzvorhersage (cadtoseq)

Ablauf
------
1. Hyperparameter-Tuning mit Optuna (Anzahl Trials in config/cadtoseq.yaml)
2. Finales Training mit den besten Hyperparametern

Konfiguration
-------------
Alle Parameter werden aus config/cadtoseq.yaml (+ config/base.yaml) gelesen.
Zum Anpassen einfach die YAML-Dateien bearbeiten – kein Code-Änderung nötig.

Ausführung
----------
    python -m mpp.ml.pipelines.cadtoseq.model_input_to_tuned_model
"""

from pathlib import Path

import mlflow
import torch

from mpp.ml.callbacks.artifact_callbacks import BestModelPlotCallback, SequencePredictionPlotCallback
from mpp.ml.models.sequence.cadtoseq_module import ARMSTM
from mpp.constants import PATHS
from mpp.ml.pipelines.base_pipeline import (
    build_callbacks,
    build_mlflow_logger,
    build_trainer,
    get_dataloaders,
    load_config,
    run_tuning,
    suggest_hyperparams,
)

_CFG_PATH = PATHS.CONFIG / "cadtoseq.yaml"
cfg = load_config(_CFG_PATH)


def main():
    train_loader, val_loader = get_dataloaders(cfg)

    # ------------------------------------------------------------------
    # Hyperparameter-Tuning
    # ------------------------------------------------------------------
    def objective(trial):
        hp = suggest_hyperparams(trial, cfg["hyperparameter_search"])
        max_epochs = cfg["training"]["tuning_epochs"]
        torch.set_float32_matmul_precision('medium')

        model = ARMSTM(
            lr=hp["lr"],
            embed_dim=hp["embed_dim"],
            nhead=hp["nhead"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
            weight_decay=cfg["training"]["weight_decay"],
            max_epochs=max_epochs,
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
            callbacks.append(SequencePredictionPlotCallback(
                plot_every_n_epochs=cfg["training"]["plot_every_n_epochs"],
            ))
            callbacks.append(BestModelPlotCallback())
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
    torch.set_float32_matmul_precision('high')

    model = ARMSTM(
        lr=best["lr"],
        embed_dim=best["embed_dim"],
        nhead=best["nhead"],
        num_layers=best["num_layers"],
        dropout=best["dropout"],
        weight_decay=cfg["training"]["weight_decay"],
        max_epochs=cfg["training"]["final_epochs"],
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
    callbacks.append(SequencePredictionPlotCallback(
        plot_every_n_epochs=cfg["training"]["plot_every_n_epochs"],
    ))
    callbacks.append(BestModelPlotCallback())
    trainer = build_trainer(cfg, cfg["training"]["final_epochs"], mlf_logger, callbacks)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
