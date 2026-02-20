"""
Pipeline: Prozesszeit-Regression (process-time-regression)

Ablauf
------
1. Hyperparameter-Tuning mit Optuna (Anzahl Trials in config/process_time_regression.yaml)
2. Finales Training mit den besten Hyperparametern

Konfiguration
-------------
Alle Parameter werden aus config/process_time_regression.yaml (+ config/base.yaml) gelesen.
Zum Anpassen einfach die YAML-Dateien bearbeiten – kein Code-Änderung nötig.

Ausführung
----------
    python -m "mpp.ml.pipelines.process-time-regression.model_input_to_tuned_model"
"""

from pathlib import Path

import mlflow

from mpp.ml.models.regressor.process_time_regressor import ProcessRegressionModule
from mpp.ml.pipelines.base_pipeline import (
    build_callbacks,
    build_mlflow_logger,
    build_trainer,
    get_dataloaders,
    load_config,
    run_tuning,
    suggest_hyperparams,
)

_CFG_PATH = Path(__file__).parents[4] / "config" / "process_time_regression.yaml"
cfg = load_config(_CFG_PATH)


def main():
    train_loader, val_loader = get_dataloaders(cfg)

    # ------------------------------------------------------------------
    # Hyperparameter-Tuning
    # ------------------------------------------------------------------
    def objective(trial):
        hp = suggest_hyperparams(trial, cfg["hyperparameter_search"])
        max_epochs = cfg["training"]["tuning_epochs"]

        model = ProcessRegressionModule(
            lr=hp["lr"],
            embed_dim=hp["embed_dim"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
            weight_decay=cfg["training"]["weight_decay"],
            max_epochs=max_epochs,
        )

        mlf_logger = build_mlflow_logger(cfg, cfg["mlflow"]["tuning_experiment_name"])
        callbacks = build_callbacks(
            cfg,
            cfg["checkpoint"]["tuning_subdir"],
            cfg["checkpoint"]["filename"],
            patience=cfg["training"]["tuning_patience"],
        )
        trainer = build_trainer(cfg, max_epochs, mlf_logger, callbacks)
        trainer.fit(model, train_loader, val_loader)

        val_loss = trainer.callback_metrics["val_loss"].item()
        mlf_logger.log_hyperparams(trial.params)
        mlf_logger.log_metrics({"val_loss": val_loss})
        return val_loss

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    study = run_tuning(cfg, objective)

    # ------------------------------------------------------------------
    # Finales Training mit besten Hyperparametern
    # ------------------------------------------------------------------
    best = study.best_trial.params

    model = ProcessRegressionModule(
        lr=best["lr"],
        embed_dim=best["embed_dim"],
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
    trainer = build_trainer(cfg, cfg["training"]["final_epochs"], mlf_logger, callbacks)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
