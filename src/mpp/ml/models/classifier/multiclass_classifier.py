import torch
import torch.nn as nn
import torch.nn.functional as F

#custom imports
from cadtoseq.constants import VOCAB
from cadtoseq.ml.datasets.fabricad import Fabricad

class TransformerProcessClassifier(nn.Module):
    def __init__(self, input_dim=32, embed_dim=512, num_heads=8, num_layers=4, dropout=0.1, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):  # x: [B, set_size, input_dim]
        x = self.embedding(x)  # [B, set_size, embed_dim]
        x = self.encoder(x)    # [B, set_size, embed_dim]
        x = x.mean(dim=1)      # [B, embed_dim] — simple average pooling
        logits = self.classifier(x)
        return logits
    

#test the model
if __name__ == "__main__":
    batch_size = 1
    vector_set = torch.randn(batch_size, 1024, 32)

    model = TransformerProcessClassifier()
    predicted_cls =model(vector_set)

    print(predicted_cls.shape)  # Expected output: [batch_size, num_classes]
    print(predicted_cls)  # Print the predicted class logits



    