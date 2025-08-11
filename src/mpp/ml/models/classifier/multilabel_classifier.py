#standard imports
import logging

#third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

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



class MultilabelTransformerEncoderModule(nn.Module):
    def __init__(self, input_dim=32, embed_dim=512, num_heads=8, num_layers=4, dropout=0.1, num_classes=2, threshold=0.5):
        """
        Transformer-based classifier for manufacturing process classification.
        Args:
            input_dim (int): Dimension of the input vector set. (is fixed by the used encoder)
            embed_dim (int): Dimension of the embedding space.
            num_heads (int): Number of attention heads in the transformer.
            num_layers (int): Number of transformer layers.
            dropout (float): Dropout rate for regularization.
            num_classes (int): Number of output classes for classification.
            threshold (float): Threshold for binary classification.
        """
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.threshold = threshold
        self.num_classes = num_classes

    def forward(self, x):  # x: [B, set_size, input_dim]
        x = self.embedding(x)  # [B, set_size, embed_dim]
        x = self.encoder(x)    # [B, set_size, embed_dim]
        x = x.mean(dim=1)      # [B, embed_dim] — avg pooling over the "Channels"
        logits = self.classifier(x)

        return logits
    
    def predict(self, x):
        logits = self.forward(x)
        if self.num_classes == 2:
            #binary classification
            probs = torch.sigmoid(logits)
            return (probs > self.threshold).float()
        else:
            #multi-class classification
            probs = torch.sigmoid(logits)           # [B, num_classes]
            return (probs > self.threshold).long()  

class ProcessClassificationTrsfmEncoderModule(pl.LightningModule):
    """
    PyTorch Lightning Module for training a vector set-based multiclass classifier.

    This module maps a set of input vectors (e.g. from CAD data) to a single class label.
    It uses mean-pooling over the input set, followed by a multilayer perceptron (MLP).
    
    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of the input vectors (default: 32).
    embed_dim : int, optional
        Size of the embedded dimension.
    num_heads : int, optional
        Number of parrallel attention-heads.
    num_layers : int, optional
        Number of layers of the trsfm-blocks.
    num_classes : int, optional
        Number of target classes (default: len(VOCAB)).
    dropout : float, optional
        Dropout probability (default: 0.3).
    lr : float, optional
        Learning rate for optimizer (default: 1e-4).
    weight_decay : float, optional
        Weight decay for optimizer regularization (default: 0.01).
    """

    def __init__(
        self,
        input_dim=32,
        embed_dim = 512,
        num_heads = 4,
        num_layers = 2,
        num_classes=len(VOCAB)-3, #-3 due to STOP, PAD, START which are no process steps
        dropout=0.3,
        lr=1e-4,
        weight_decay=0.01,
        threshold = 0.5,
        max_epochs = 50
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = MultilabelTransformerEncoderModule(input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout, num_classes=num_classes, threshold=threshold)

        if num_classes == 2:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.max_epochs = max_epochs

    def forward(self, x):
        """
        Forward pass of the classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, set_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        #pooled = x.mean(dim=1)  # Mean pooling over set dimension
        logger.debug(f"Forward input: {x.shape}")
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Training step for one batch.

        Parameters
        ----------
        batch : tuple
            Tuple of (vector_set, class_label).
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        x, y = batch
        assert y.dim() == 2, f"Expected y shape (B, C), got {y.shape}"
        assert y.dtype == torch.float, f"Expected y dtype float, got {y.dtype}"

        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Optional: Accuracy (thresholded sigmoid output)
        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y.bool()).float().mean()

        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch.

        Parameters
        ----------
        batch : tuple
            Tuple of (vector_set, class_label).
        batch_idx : int
            Index of the current validation batch.
        """
        x, y = batch
        assert y.dim() == 2, f"Expected y shape (B, C), got {y.shape}"
        assert y.dtype == torch.float, f"Expected y dtype float, got {y.dtype}"

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.sigmoid(logits) > 0.5
        acc = (preds == y.bool()).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary containing optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=200, 
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    

#test the model
if __name__ == "__main__":
    batch_size = 16
    vector_set = torch.randn(batch_size, 1024, 32)

    model = MultilabelTransformerEncoderModule(num_classes=10)
    model.eval()  
    logits =model(vector_set)

    predicted_cls = model.predict(vector_set)

    print(predicted_cls.shape)  
    print(predicted_cls)  



    