"""
Gemeinsame Pipeline-Utilities für alle MPP-Trainingsaufgaben.

Jede Task-Pipeline (cadtoseq, cadtostepset, process-time-regression) nutzt
diese Funktionen, um Dataloaders, MLflow-Logger, Callbacks und Trainer zu
erstellen. Die Konfiguration erfolgt über YAML-Dateien im config/-Verzeichnis.

Verwendung
----------
    from mpp.ml.pipelines.base_pipeline import load_config, get_dataloaders, ...

    cfg = load_config(Path("config/cadtoseq.yaml"))
    train_loader, val_loader = get_dataloaders(cfg)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import mlflow
import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from mpp.constants import PATHS
from mpp.ml.callbacks.artifact_callbacks import MLflowCheckpointCallback
from mpp.ml.datasets.fabricad_datamodule import Fabricad_datamodule

_BASE_CONFIG_PATH = PATHS.CONFIG / "base.yaml"


# ---------------------------------------------------------------------------
# Config-Loading
# ---------------------------------------------------------------------------

def load_config(task_config_path: str | Path) -> dict[str, Any]:
    """Lade und merge base.yaml mit einer Task-spezifischen Config-Datei.

    Die Task-Config überschreibt Werte der Basis-Config (deep merge).

    Parameters
    ----------
    task_config_path:
        Pfad zur aufgaben-spezifischen YAML-Datei.

    Returns
    -------
    dict
        Zusammengeführte Konfiguration.
    """
    with open(_BASE_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    with open(task_config_path) as f:
        task_cfg = yaml.safe_load(f)
    return _deep_merge(cfg, task_cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    """Rekursiver Deep-Merge: override-Werte überschreiben base-Werte."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: dict):
    """Erstelle und liefere (train_loader, val_loader) aus der Config.

    Parameters
    ----------
    cfg:
        Zusammengeführte Konfiguration (Ergebnis von load_config()).

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training- und Validierungs-DataLoader.
    """
    dm = Fabricad_datamodule(
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        input_type=cfg["data"]["input_type"],
        target_type=cfg["data"]["target_type"],
    )
    dm.setup(stage="fit")
    return dm.train_dataloader(), dm.val_dataloader()


# ---------------------------------------------------------------------------
# MLflow-Logger
# ---------------------------------------------------------------------------

def build_mlflow_logger(
    cfg: dict,
    experiment_name: str,
    run_name: str | None = None,
    run_id: str | None = None,
) -> MLFlowLogger:
    """Erstelle einen MLFlowLogger mit der konfigurierten Tracking-URI.

    Parameters
    ----------
    cfg:
        Zusammengeführte Konfiguration.
    experiment_name:
        Name des MLflow-Experiments.
    run_name:
        Optionaler Name des MLflow-Runs.
    run_id:
        Optionale Run-ID eines bereits gestarteten MLflow-Runs (z. B. für
        Nested Runs).

    Returns
    -------
    MLFlowLogger
    """
    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=cfg["mlflow"]["tracking_uri"],
        run_name=run_name,
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def build_callbacks(
    cfg: dict,
    checkpoint_subdir: str,
    filename: str,
    patience: int,
) -> list:
    """Erstelle [EarlyStopping, ModelCheckpoint] Callbacks.

    Parameters
    ----------
    cfg:
        Zusammengeführte Konfiguration.
    checkpoint_subdir:
        Unterverzeichnis relativ zu PATHS.CKPT_DIR für Checkpoints.
    filename:
        Dateinamenmuster für ModelCheckpoint.
    patience:
        Patience für EarlyStopping.

    Returns
    -------
    list
        [EarlyStopping, ModelCheckpoint]
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        dirpath=(PATHS.CKPT_DIR / checkpoint_subdir).as_posix(),
        filename=filename,
        save_weights_only=False,
        verbose=True,
    )
    return [early_stop, checkpoint, MLflowCheckpointCallback()]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def build_trainer(
    cfg: dict,
    max_epochs: int,
    logger: MLFlowLogger,
    callbacks: list,
) -> Trainer:
    """Erstelle einen PyTorch Lightning Trainer.

    Parameters
    ----------
    cfg:
        Zusammengeführte Konfiguration.
    max_epochs:
        Maximale Anzahl Epochen.
    logger:
        MLFlowLogger-Instanz.
    callbacks:
        Liste von Lightning-Callbacks.

    Returns
    -------
    Trainer
    """
    return Trainer(
        max_epochs=max_epochs,
        precision="bf16-mixed",
        logger=logger,
        enable_checkpointing=True,
        enable_model_summary=False,
        log_every_n_steps=cfg["training"]["log_every_n_steps"],
        callbacks=callbacks,
        devices=1
    )


# ---------------------------------------------------------------------------
# Optuna Tuning
# ---------------------------------------------------------------------------

def run_tuning(cfg: dict, objective) -> optuna.Study:
    """Führe Optuna-Hyperparameter-Suche durch und gib die Study zurück.

    Parameters
    ----------
    cfg:
        Zusammengeführte Konfiguration.
    objective:
        Optuna-Zielfunktion (trial) -> float.

    Returns
    -------
    optuna.Study
        Abgeschlossene Optuna-Study mit allen Trials.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=cfg["training"]["n_trials"])
    return study


# ---------------------------------------------------------------------------
# Hilfsfunktion: Hyperparameter aus Config samplen
# ---------------------------------------------------------------------------

def suggest_hyperparams(trial: optuna.Trial, hp: dict) -> dict:
    """Sample Hyperparameter aus dem konfigurierten Suchraum.

    Unterstützte Typen pro Parameter:
    - float: {low, high, log?}
    - int:   {low, high, step?}
    - categorical: {choices}

    Parameters
    ----------
    trial:
        Optuna-Trial.
    hp:
        Hyperparameter-Suchraum aus cfg["hyperparameter_search"].

    Returns
    -------
    dict
        Gesamplete Hyperparameter {name: value}.
    """
    params: dict[str, Any] = {}
    for name, spec in hp.items():
        if "choices" in spec:
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif isinstance(spec.get("low"), float) or spec.get("log", False):
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
        else:
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
            )
    return params
