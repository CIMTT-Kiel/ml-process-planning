import logging

import torch
import torch.nn as nn

from mpp.constants import VOCAB

logger = logging.getLogger(__name__)


class MultilabelTransformerEncoderModule(nn.Module):
    """Transformer-basierter Encoder für Multi-Label-Klassifikation.

    Bildet einen Satz von Eingabevektoren (z.B. aus CAD-Daten) auf
    Klassen-Logits ab. Das Pooling erfolgt über den Set-Dim (mean).

    Parameters
    ----------
    input_dim : int
        Dimension der Eingabevektoren (Standard: 32, fix durch den Encoder).
    embed_dim : int
        Dimension des Embedding-Raums.
    num_heads : int
        Anzahl der Attention-Heads.
    num_layers : int
        Anzahl der Transformer-Encoder-Schichten.
    dropout : float
        Dropout-Rate.
    num_classes : int
        Anzahl der Ausgabe-Klassen.
    threshold : float
        Schwellenwert für die binäre Klassifikation (predict()).
    """

    def __init__(
        self,
        input_dim: int = 32,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.threshold = threshold
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-Pass.

        Parameters
        ----------
        x : torch.Tensor
            Eingabe der Form (B, set_size, input_dim).

        Returns
        -------
        torch.Tensor
            Logits der Form (B, num_classes).
        """
        x = self.embedding(x)   # [B, set_size, embed_dim]
        x = self.encoder(x)     # [B, set_size, embed_dim]
        x = x.mean(dim=1)       # [B, embed_dim] – avg pooling über Set-Dim
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Schwellenwert-basierte Vorhersage.

        Parameters
        ----------
        x : torch.Tensor
            Eingabe der Form (B, set_size, input_dim).

        Returns
        -------
        torch.Tensor
            Binäre Vorhersagen der Form (B, num_classes).
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        if self.num_classes == 2:
            return (probs > self.threshold).float()
        return (probs > self.threshold).long()


if __name__ == "__main__":
    batch_size = 16
    vector_set = torch.randn(batch_size, 1024, 32)

    model = MultilabelTransformerEncoderModule(num_classes=10)
    model.eval()
    logits = model(vector_set)
    predicted_cls = model.predict(vector_set)

    print(predicted_cls.shape)
    print(predicted_cls)
