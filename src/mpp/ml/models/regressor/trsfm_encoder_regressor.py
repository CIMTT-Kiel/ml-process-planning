import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TrsfmEncoderRegressor(nn.Module):
    """
    Transformer-Encoder-basiertes Regressionsmodell für Fertigungsprozesszeiten.

    Parameters
    ----------
    input_dim : int
        Dimensionalität der Eingabevektoren im Vector Set.
    embed_dim : int
        Größe des Embedding-Raums im Transformer.
    num_heads : int
        Anzahl der Attention Heads.
    num_layers : int
        Anzahl der Transformer-Encoder-Layer.
    dropout : float
        Dropout-Rate.
    """

    def __init__(self, input_dim=32, embed_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Linear(embed_dim, 1)

    def forward(self, x):  # x: [B, set_size, input_dim]
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)           # [B, embed_dim] — permutation-invariantes Pooling
        return self.regressor(x).squeeze(-1)  # [B]


if __name__ == "__main__":
    batch_size = 16
    vector_set = torch.randn(batch_size, 1024, 32)

    model = TrsfmEncoderRegressor()
    model.eval()
    output = model(vector_set)

    print(output.shape)  # should be [B]
    print(output)
