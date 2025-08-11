#standard library imports

# third party imports
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn

#custom imports
from mpp.constants import VOCAB
from mpp.ml.models.decoder.vecset_transformer import ARMSTD
from mpp.ml.metrics.sequences import Sequence_comparator

class ARMSTM(pl.LightningModule):
    """
    PyTorch Lightning Module for training a Transformer-based model for 
    autoregressive manufacturing step prediction.

    This module wraps the ARMSTD decoder model and provides:
    - Autoregressive sequence modeling using teacher forcing during training
    - Training, validation, and inference logic
    - Cross-entropy loss (with PAD-token masking)
    - Sequence-level evaluation metrics (exact match, Levenshtein distance, etc.)
    - Learning rate scheduling and optimizer configuration

    Parameters
    ----------
    lr : float, optional
        Learning rate for the optimizer (default: 3e-5).
    embed_dim : int, optional
        Embedding dimension used in the transformer (default: 128).
    nhead : int, optional
        Number of attention heads in the transformer (default: 4).
    num_layers : int, optional
        Number of transformer decoder layers (default: 3).
    dropout : float, optional
        Dropout probability (default: 0.3).
    weight_decay : float, optional
        Weight decay for regularization (default: 0.01).
    max_epochs : int, optional
        Maximum number of epochs for training (default: 100).
    """

    def __init__(self, lr=0.00003, embed_dim=128, nhead=4, num_layers = 3, dropout=0.3,  weight_decay=0.01, max_epochs=100):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        #model spezific
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.model = ARMSTD(embed_dim=self.embed_dim, num_layers=self.num_layers, nhead=self.nhead, dropout=self.dropout)
        
        self.save_hyperparameters()
        self.save_hyperparameters("lr", "embed_dim", "nhead", "num_layers", "dropout", "weight_decay", "max_epochs")

        self.criterion = nn.CrossEntropyLoss(ignore_index=VOCAB["PAD"])  # ignore PAD tokens in the loss calculation

        # Initialize the sequence comparator to evaluate additional sequence metrics
        self.comparator = Sequence_comparator(VOCAB)


    def forward(self, vector_set, tgt_seq):
        """
        Forward pass of the model.

        Parameters
        ----------
        vector_set : torch.Tensor
            Input features of shape (batch_size, set_size, input_dim).
        tgt_seq : torch.Tensor
            Target sequence input to the decoder (e.g., shifted ground truth tokens).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, seq_len, num_classes).
        """

        
        return self.model(vector_set, tgt_seq)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step using teacher forcing.

        Parameters
        ----------
        batch : tuple
            A batch containing input vector sets and padded target sequences.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            The computed training loss.
        """
        vector_set, padded_targets = batch

        decoder_input = torch.full((padded_targets.size(0), 1), VOCAB["START"], dtype=torch.long).to(padded_targets.device)
        decoder_input = torch.cat([decoder_input, padded_targets[:, :-1]], dim=1).to(padded_targets.device)

        logits = self(vector_set, decoder_input)

        loss = self.criterion(logits.view(-1,VOCAB.__len__()), padded_targets.view(-1))

        self.log("train_loss", loss)

        return loss
    
    def generate(self, vector_set, return_probs=False, device="cpu"):
        """
        Generate sequences in autoregressive fashion using the decoder model.

        Parameters
        ----------
        vector_set : torch.Tensor
            Input vector sets.
        return_probs : bool, optional
            Whether to return token probabilities (default: False).
        device : str, optional
            Device to run generation on (default: "cpu").

        Returns
        -------
        torch.Tensor
            Generated sequences (without START token).
        """
        return self.model.generate(vector_set, return_probs=False, device=device)

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step with optional metric logging.

        Parameters
        ----------
        batch : tuple
            A batch containing input vector sets and padded target sequences.
        batch_idx : int
            Index of the current validation batch.
        """
        vector_set, padded_targets = batch

        decoder_input = torch.full((padded_targets.size(0), 1), VOCAB["START"], dtype=torch.long).to(padded_targets.device)
        decoder_input = torch.cat([decoder_input, padded_targets[:, :-1]], dim=1).to(padded_targets.device)

        logits = self(vector_set, decoder_input)

        preds = logits.argmax(dim=-1)
        mask = padded_targets != VOCAB["PAD"]
        acc = ((preds == padded_targets) & mask).sum().float() / mask.sum()
        self.log("val_acc", acc, prog_bar=True)

        val_loss = self.criterion(logits.view(-1,VOCAB.__len__()), padded_targets.view(-1))

        self.log("val_loss", val_loss, prog_bar=True)

        # log additional metrics only every second validation step

        if batch_idx % 2 == 0:
            # Calculate additional sequence metrics
            s_metrics = self.comparator.compare(preds, padded_targets)

            self.log("val_elementwise_accuracy", s_metrics["elementwise_accuracy"].mean(), prog_bar=True)
            self.log("val_shifted_accurracy", s_metrics["shifted_accuracy"].mean(), prog_bar=True)
            self.log("val_exact_match", s_metrics["exact_match"].to(torch.float).mean(), prog_bar=True)
            self.log("val_levenshtein_distance", s_metrics["levenshtein_distance"].mean(), prog_bar=True)


        # Log the learning rate
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])

        
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary with optimizer and scheduler configuration.
        """
        # AdamW-Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),          
            lr=self.hparams.lr,        
            weight_decay=self.hparams.weight_decay  
        )

        # Cosine Annealing Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=200,  # max epochs for one cycle
            eta_min=0.000001  # min learning rate to reach
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss" 
            }
        }