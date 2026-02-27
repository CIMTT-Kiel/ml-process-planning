#standard library imports
import logging

#third party imports
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

#custom imports
from mpp.constants import VOCAB
from mpp.ml.datasets.fabricad import Fabricad

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")

def collate_fn(batch):
    """
    Custom collate function for batching variable-length target sequences.

    This function stacks input feature vectors and pads the target sequences
    (plans) to a fixed maximum length using the PAD token from the VOCAB.

    Parameters
    ----------
    batch : list of tuples
        Each element is a (vecset, plan) tuple where:
        - vecset : torch.Tensor of shape (set_size, input_dim)
        - plan : torch.Tensor of variable length (sequence of token indices)

    Returns
    -------
    vecsets : torch.Tensor
        Stacked feature vectors of shape (batch_size, set_size, input_dim).
    padded_plans : torch.Tensor
        Padded sequences of shape (batch_size, max_len), where padding tokens
        are added to match the maximum allowed sequence length.
    """
    vecsets, plans = zip(*batch)
    vecsets = torch.stack(vecsets)

    max_len = 10
    padded_plans = torch.full((len(plans), max_len), VOCAB["PAD"], dtype=torch.long)

    for i, plan in enumerate(plans):
        padded_plans[i, :plan.size(0)] = plan

    return vecsets, padded_plans


def collate_fn_step_time(batch):
    """Custom collate function für den ``"step-time"``-Target-Typ.

    Stapelt Geometrie-Embeddings und paddet Token- und Zeitsequenzen variabler
    Länge auf die längste Sequenz im Batch.

    Parameters
    ----------
    batch : list of tuples
        Jedes Element ist ein ``(vecset, (step_tokens, step_times, total_time))``-
        Tupel, wie es von :class:`~mpp.ml.datasets.fabricad.Fabricad` mit
        ``target_type="step-time"`` zurückgegeben wird.

    Returns
    -------
    vecsets : torch.Tensor
        Shape ``[B, set_size, input_dim]``.
    padded_tokens : torch.Tensor
        Shape ``[B, max_seq_len]``, PAD-Stellen mit ``VOCAB["PAD"]`` gefüllt.
    padded_times : torch.Tensor
        Shape ``[B, max_seq_len]``, PAD-Stellen mit ``0.0`` gefüllt.
    total_times : torch.Tensor
        Shape ``[B]``, Gesamtdauer je Probe (Summe der gefilterten Schritte).
    """
    vecsets, targets = zip(*batch)
    step_tokens_list, step_times_list, total_times = zip(*targets)

    vecsets = torch.stack(vecsets)                            # [B, set_size, input_dim]
    total_times = torch.stack(list(total_times))              # [B]

    max_len = max(t.size(0) for t in step_tokens_list)
    B = len(step_tokens_list)

    padded_tokens = torch.full((B, max_len), VOCAB["PAD"], dtype=torch.long)
    padded_times = torch.zeros(B, max_len, dtype=torch.float32)

    for i, (tokens, times) in enumerate(zip(step_tokens_list, step_times_list)):
        n = tokens.size(0)
        padded_tokens[i, :n] = tokens
        padded_times[i, :n] = times

    return vecsets, padded_tokens, padded_times, total_times

def collate_fn_mtl(batch):
    """Custom collate function für den ``"step-time-cost"``-Target-Typ.

    Stapelt Geometrie-Embeddings und paddet Token-, Zeit- und Kostensequenzen
    variabler Länge auf die längste Sequenz im Batch.

    Parameters
    ----------
    batch : list of tuples
        Jedes Element ist ein
        ``(vecset, (step_tokens, step_times, step_costs, total_time, total_cost))``-
        Tupel, wie es von :class:`~mpp.ml.datasets.fabricad.Fabricad` mit
        ``target_type="step-time-cost"`` zurückgegeben wird.

    Returns
    -------
    vecsets : torch.Tensor
        Shape ``[B, set_size, input_dim]``.
    padded_tokens : torch.Tensor
        Shape ``[B, max_seq_len]``, PAD-Stellen mit ``VOCAB["PAD"]`` gefüllt.
    padded_times : torch.Tensor
        Shape ``[B, max_seq_len]``, PAD-Stellen mit ``0.0`` gefüllt.
    padded_costs : torch.Tensor
        Shape ``[B, max_seq_len]``, PAD-Stellen mit ``0.0`` gefüllt.
    total_times : torch.Tensor
        Shape ``[B]``, Gesamtdauer je Probe.
    total_costs : torch.Tensor
        Shape ``[B]``, Gesamtkosten je Probe.
    """
    vecsets, targets = zip(*batch)
    step_tokens_list, step_times_list, step_costs_list, total_times, total_costs = zip(*targets)

    vecsets = torch.stack(vecsets)                                    # [B, set_size, input_dim]
    total_times = torch.stack(list(total_times))                      # [B]
    total_costs = torch.stack(list(total_costs))                      # [B]

    max_len = max(t.size(0) for t in step_tokens_list)
    B = len(step_tokens_list)

    padded_tokens = torch.full((B, max_len), VOCAB["PAD"], dtype=torch.long)
    padded_times = torch.zeros(B, max_len, dtype=torch.float32)
    padded_costs = torch.zeros(B, max_len, dtype=torch.float32)

    for i, (tokens, times, costs) in enumerate(zip(step_tokens_list, step_times_list, step_costs_list)):
        n = tokens.size(0)
        padded_tokens[i, :n] = tokens
        padded_times[i, :n] = times
        padded_costs[i, :n] = costs

    return vecsets, padded_tokens, padded_times, padded_costs, total_times, total_costs


class Fabricad_datamodule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading the Fabricad dataset.

    This module handles loading, splitting, batching, and preprocessing
    of the Fabricad dataset according to the specified input and target types.

    Parameters
    ----------
    batch_size : int, optional
        Batch size to be used in data loaders (default: 32).
    num_workers : int, optional
        Number of subprocesses to use for data loading (default: 4).
    input_type : str, optional
        Type of input to be used. Options include:
        - "vecset": for vector set input (e.g. CAD representations)
        - Other types may be supported depending on the dataset implementation.
    target_type : str, optional
        Type of target labels. Options include:
        - "seq": for step-by-step sequences
        - "class": for single-label classification
        - Others as defined in the Fabricad dataset.
    """
    def __init__(self, batch_size=32, num_workers=0, input_type="vecset", target_type="seq"):
        super().__init__()
        logger.info("Initializing Fabricad datamodule")
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.input_type=input_type
        self.target_type =target_type

    def setup(self, stage=None):
        """
        Sets up datasets for different stages of training, validation, and testing.

        Parameters
        ----------
        stage : str or None
            One of 'fit', 'test', or None (default).
            If 'fit', initializes training and validation datasets.
            If 'test', initializes the test dataset.
            If None, initializes all datasets.

        Notes
        -----
        The setup uses the configured input and target types to load the data accordingly.
        These are passed directly to the `Fabricad` dataset constructor.
        """
        logger.info(f"Setting up Fabricad datamodule for stage: {stage}")
        if stage == "fit" or stage is None:
            self.train_dataset = Fabricad(mode="train", input_type=self.input_type, target_type=self.target_type)
            self.val_dataset = Fabricad(mode="valid", input_type=self.input_type, target_type=self.target_type)
            logger.info(f"Train dataset size: {len(self.train_dataset)}, Validation dataset size: {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = Fabricad(mode="test", input_type=self.input_type, target_type=self.target_type)
            logger.info(f"Test dataset size: {len(self.test_dataset)}")

    def _get_collate_fn(self):
        """Gibt die passende collate-Funktion für den konfigurierten target_type zurück."""
        if self.target_type == "seq":
            return collate_fn
        if self.target_type == "step-time":
            return collate_fn_step_time
        if self.target_type == "step-time-cost":
            return collate_fn_mtl
        return None  # Standard-collate von PyTorch

    def train_dataloader(self):
        """
        Returns the training data loader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for the training set.
        """
        logger.debug("Creating train dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
        )

    def val_dataloader(self):
        """
        Returns the validation data loader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for the validation set.
        """
        logger.debug("Creating validation dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
        )

    def test_dataloader(self):
        """
        Returns the test data loader.

        Returns
        -------
        DataLoader
            PyTorch DataLoader for the test set.
        """
        logger.debug("Creating test dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
        )



#checks
if __name__ == "__main__":
    
    for input_type in ["vecset"]:
        for target_type in ["time", "cost", "step-set", "seq"]:
            vecset_data_module = Fabricad_datamodule(batch_size=32, num_workers=0, input_type=input_type, target_type=target_type)
            vecset_data_module.setup(stage="fit")
            
            train_loader = vecset_data_module.train_dataloader()
            validation_loader = vecset_data_module.val_dataloader()

            train_batch = next(iter(train_loader))
            validation_batch = next(iter(validation_loader))

            logger.info("Train batch shape:", train_batch[0].shape, train_batch[1].shape)
            logger.info("Validation batch shape:", validation_batch[0].shape, validation_batch[1].shape)


