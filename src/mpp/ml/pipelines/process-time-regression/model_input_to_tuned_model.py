# third party imports
import optuna
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# custom imports
from src.mpp.ml.models.regressor.trsfm_encoder_regressor import ProcessRegressionModule  # dein Classifier-Modul
from mpp.ml.datasets.fabricad_datamodule import Fabricad_datamodule
from mpp.constants import PATHS


def get_dataloaders(batch_size=32, input_type="vecset", target_type="time"):
    """
    Initializes and returns the training and validation dataloaders for classification task.

    Parameters
    ----------
    batch_size : int, optional
        Number of samples per batch to load. Default is 32.
    input_type : str, optional
        Input data format type. Default "vecset".
    target_type : str, optional
        Target data format type, e.g. "tmie" for classification labels. Default "time".

    Returns
    -------
    tuple of torch.utils.data.DataLoader
        A tuple containing (train_loader, validation_loader).
    """
    dataset = Fabricad_datamodule(batch_size=batch_size, input_type=input_type, target_type=target_type)
    dataset.setup(stage="fit")

    train_loader = dataset.train_dataloader()
    validation_loader = dataset.val_dataloader()

    return train_loader, validation_loader


batch_size = 85
train_loader, val_loader = get_dataloaders(batch_size=batch_size)


def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization on regression model.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object used to suggest hyperparameters.

    Returns
    -------
    float
        Validation loss to minimize.
    """
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    # Beispielhafte Hyperparameter für den Classifier (anpassen nach deinem Modell)
    embed_dim = trial.suggest_int("embed_dim", 64, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 2, 6)

    #hard coded
    max_epochs=20

    model = ProcessRegressionModule(
        lr=lr,
        embed_dim=embed_dim,
        num_layers=num_layers,
        dropout=dropout,
        weight_decay=0.01,
        max_epochs=max_epochs
    )

    mlf_logger = MLFlowLogger(experiment_name="process-time-regression-hyperparameter-tuning", tracking_uri="file:./mlruns")

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath=(PATHS.CKPT_DIR / "tuning/time-regression").as_posix(),
        filename='time-regressor-{epoch:02d}-{val_loss:.4f}',
        save_weights_only=False,
        verbose=True
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=mlf_logger,
        enable_checkpointing=True,
        enable_model_summary=False,
        log_every_n_steps=2,
        callbacks=[early_stop_callback, checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)
    val_loss = trainer.callback_metrics["val_loss"].item()

    mlf_logger.log_hyperparams(trial.params)
    mlf_logger.log_metrics({"val_loss": val_loss})

    return val_loss


def main():
    """
    Main function to run Optuna tuning and train best classification model.
    """
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("classification")

    mlf_logger = MLFlowLogger(experiment_name="process-time-regression", tracking_uri="file:./mlruns", run_name="best-model")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    best_params = study.best_trial.params

    model = ProcessRegressionModule(
        lr=best_params["lr"],
        embed_dim=best_params["embed_dim"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
        weight_decay=0.01,
        max_epochs=1000
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=30,
        mode='min',
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        dirpath=(PATHS.CKPT_DIR / "best_model/time-regression").as_posix(),
        filename='time-regressor-{epoch:02d}-{val_loss:.4f}',
        save_weights_only=False,
        verbose=True
    )

    trainer = Trainer(
        max_epochs=1000,
        logger=mlf_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_checkpointing=True,
        enable_model_summary=False,
        log_every_n_steps=1
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
