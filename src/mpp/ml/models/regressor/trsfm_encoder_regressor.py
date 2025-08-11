# standard library imports
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
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")


class TrsfmEncoderRegressor(nn.Module):
    def __init__(self, input_dim=32, embed_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        """
        Regressor model for manufacturing process prediction.
        Args:
            input_dim (int): Dimension of the input vector set.
            embed_dim (int): Dimension of the embedding space.
            num_heads (int): Number of attention heads in the transformer.
            num_layers (int): Number of transformer layers.
            dropout (float): Dropout rate for regularization.
        """
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Linear(embed_dim, 1)  # Ausgabe: ein Skalar

    def forward(self, x):  # x: [B, set_size, input_dim]
        x = self.embedding(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        output = self.regressor(x)  # [B, 1]
        return output.squeeze(-1)   # [B]


#test the model
if __name__ == "__main__":
    batch_size = 16
    vector_set = torch.randn(batch_size, 1024, 32)

    model = TrsfmEncoderRegressor()
    model.eval()  
    output = model(vector_set)

    print(output.shape)  # should be [B]
    print(output)  # should print the regression outputs