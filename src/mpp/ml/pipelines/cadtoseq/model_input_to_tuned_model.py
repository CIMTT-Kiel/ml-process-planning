#third party imports
import optuna
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


#custom imports
from src.mpp.ml.models.sequence.cadtoseq_module import ARMSTM
from mpp.ml.datasets.fabricad_datamodule import Fabricad_datamodule
from mpp.constants import PATHS



def get_dataloaders(batch_size=32):
    """
    Initializes and returns the training and validation dataloaders for the Fabricad dataset.

    This function creates an instance of the Fabricad_datamodule, sets it up for training,
    and returns the corresponding PyTorch DataLoader objects.

    Parameters
    ----------
    batch_size : int, optional
        Number of samples per batch to load. Default is 32.

    Returns
    -------
    tuple of torch.utils.data.DataLoader
        A tuple containing (train_loader, validation_loader).
    """
    # Initialize the Fabricad datamodule
    dataset = Fabricad_datamodule(batch_size=batch_size)
    dataset.setup(stage="fit")

    train_loader = dataset.train_dataloader()
    validation_loader = dataset.val_dataloader()

    return train_loader, validation_loader


batch_size = 85
train_loader, val_loader = get_dataloaders(batch_size=batch_size)

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.

    This function:
    - Samples hyperparameters from a given Optuna trial.
    - Initializes the model with these hyperparameters.
    - Sets up early stopping, checkpointing, and MLflow logging.
    - Trains the model using PyTorch Lightning's Trainer.
    - Returns the validation loss to guide the optimization.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The Optuna trial object used to suggest hyperparameters.

    Returns
    -------
    float
        The validation loss of the trained model, used as the objective to minimize.
    """
    # Hyperparameter-sampling
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    # Transformer-specific hyperparameters
    nhead = trial.suggest_categorical("nhead", [4, 8])
    num_layers = trial.suggest_int("num_layers", 4, 8)
    embed_dim = trial.suggest_int("embed_dim", 64, 256, step=32)

    model = ARMSTM( 
                lr=lr,
                embed_dim=embed_dim, 
                nhead=nhead, 
                num_layers=num_layers, 
                dropout=dropout,
                weight_decay=0.01,
                max_epochs=50
            )

    # Setup MLflow Logger
    mlf_logger = MLFlowLogger(experiment_name="cadtoseq-hyperparameter-tuning", tracking_uri="file:./mlruns")

    # Early Stopping Callback
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
        dirpath=(PATHS.CKPT_DIR / "tuning").as_posix(),
        filename='manuf_step_decoder-{epoch:02d}-{val_loss:.4f}', 
        save_weights_only=False,      
        verbose=True
    )

    # Trainer setup
    trainer = Trainer(
        max_epochs=50,
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
    Entry point for training the best model using hyperparameters optimized by Optuna.

    This function:
    - Sets up the MLflow experiment.
    - Runs Optuna to find the best hyperparameters.
    - Initializes the best model with the optimal configuration.
    - Trains the model using early stopping and checkpointing.
    - Logs training metrics and saves the best checkpoint.

    Side Effects
    ------------
    - Creates and writes to `mlruns/` directory for MLflow tracking.
    - Saves model checkpoints to the `PATHS.CKPT_DIR` directory.
    """
    mlflow.set_tracking_uri("file:./mlruns") 
    mlflow.set_experiment("cadtoseq")

    mlf_logger = MLFlowLogger(experiment_name="cadtoseq", tracking_uri="file:./mlruns", run_name="best-model")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    
    best_params = study.best_trial.params

    model = ARMSTM( 
                lr = best_params["lr"],
                embed_dim=best_params["embed_dim"],
                nhead=best_params["nhead"], 
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
        dirpath=(PATHS.CKPT_DIR / "best_model").as_posix(),
        filename='manuf_step_decoder-{epoch:02d}-{val_loss:.4f}', 
        save_weights_only=False,      
        verbose=True
    )

    trainer = Trainer(max_epochs=1000, logger=mlf_logger, callbacks=[early_stop_callback, checkpoint_callback], enable_checkpointing=True, enable_model_summary=False, log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()