#standard imports
import logging
import math

#third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

#custom imports
from mpp.constants import VOCAB
from mpp.ml.datasets.fabricad import Fabricad
from mpp.constants import VOCAB

logging.basicConfig(
    format="%(asctime)s %(levelname)8s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s %(levelname)8s - %(message)s")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, embed_dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ARMSTD(nn.Module):
    """
    AutoRegressive Manufacturing Step Transformer Decoder (ARMSTD)

    This model predicts sequences of manufacturing process steps based on an input
    set of feature vectors. It uses a Transformer decoder architecture in an autoregressive
    fashion to generate token sequences, such as process step plans.

    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of each input vector in the set (default: 32).
    set_size : int, optional
        Number of vectors in each input set (default: 1024).
    embed_dim : int, optional
        Dimensionality of the embedding space (default: 512).
    num_steps : int, optional
        Number of distinct step tokens in the vocabulary (default: len(VOCAB)).
    max_seq_len : int, optional
        Maximum length of the generated process sequence (excluding START token) (default: 6).
    num_layers : int, optional
        Number of layers in the Transformer decoder (default: 6).
    nhead : int, optional
        Number of attention heads (default: 8).
    dropout : float, optional
        Dropout rate used in various layers (default: 0.1).
    """
    def __init__(self, input_dim=32, set_size=1024, embed_dim=512, num_steps=VOCAB.__len__(), max_seq_len=6, num_layers=6, nhead=8, dropout=0.1):
        super().__init__()

        self.input_linear = nn.Linear(input_dim, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=4 * embed_dim, batch_first=True, dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.step_embeddings = nn.Embedding(num_steps, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.output_linear = nn.Linear(embed_dim, num_steps)

        self.max_seq_len = max_seq_len
        self.num_steps = num_steps
        self.stop_token_id = VOCAB["STOP"]  

        self.input_dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, vector_set, tgt_seq):
        """
        Forward pass of the Transformer decoder.

        Parameters
        ----------
        vector_set : torch.Tensor
            Input tensor of shape (batch_size, set_size, input_dim), representing feature vectors.
        tgt_seq : torch.Tensor
            Tensor of token indices representing the target sequence 
            (batch_size, seq_len), usually used during teacher forcing.

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, seq_len, num_steps), 
            which can be used with CrossEntropyLoss.
        """
        batch_size = vector_set.size(0)
        memory = self.input_dropout(self.input_linear(vector_set)) 

        tgt_embedded = self.pos_encoder(self.embedding_dropout(self.step_embeddings(tgt_seq)))
        tgt_key_padding_mask = tgt_seq == VOCAB["PAD"]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_embedded.size(1), device=vector_set.device
        ).bool()

        output = self.decoder(
            tgt=tgt_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.output_linear(output)
        return logits

    def generate(self, vector_set, return_probs=False, device="cpu"):
        """
        Autoregressively generates a sequence of manufacturing steps given input vectors.

        Starts with a START token and generates one token at a time until a STOP token
        is produced or the maximum sequence length is reached.

        Parameters
        ----------
        vector_set : torch.Tensor
            Input tensor of shape (batch_size, set_size, input_dim), representing feature vectors.
        return_probs : bool, optional
            If True, returns softmax probabilities for each step (default: False).
        device : str, optional
            Device to perform computation on (e.g., 'cpu' or 'cuda').

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If return_probs is False:
                Generated token sequences of shape (batch_size, max_seq_len)
            If return_probs is True:
                Tuple of:
                    - Generated sequences (batch_size, max_seq_len)
                    - Softmax probabilities (batch_size, max_seq_len, num_steps)
        """
        batch_size = vector_set.size(0)
        vector_set = vector_set.to(device)
        memory = self.input_linear(vector_set) 

        generated = torch.full((batch_size, 1), VOCAB["START"], dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        all_probs = []

        for _ in range(self.max_seq_len):
            tgt_embedded = self.pos_encoder(self.step_embeddings(generated))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_embedded.size(1), device=device
            ).bool()

            output = self.decoder(
                tgt=tgt_embedded,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_is_causal=True,
            )

            logits = self.output_linear(output[:, -1, :])
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.unsqueeze(1))

            next_token = torch.argmax(probs, dim=-1).to(device)
            
            next_token[finished] = VOCAB["PAD"]

            finished |= (next_token == self.stop_token_id)

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # add padding if for one or more sequences the stop token was reached or 
        if generated.size(1) < self.max_seq_len + 1:  # the plus one is because of the manual added START token
            pad_len = self.max_seq_len + 1 - generated.size(1)
            pad = torch.full((batch_size, pad_len), VOCAB["PAD"], dtype=torch.long, device=device)
            generated = torch.cat([generated, pad], dim=1)

        all_probs = torch.cat(all_probs, dim=1) if return_probs else None

        # Remove the START token from the generated sequence
        return (generated[:, 1:], all_probs) if return_probs else generated[:, 1:]
    
    def train_model():
        pass  


if __name__ == "__main__":
    batch_size = 1
    vector_set = torch.randn(batch_size, 1024, 32)

    model = ARMSTD()
    generated_seq = model.generate(vector_set, device=vector_set.device)

    print(Fabricad.decode_sequence(generated_seq[0].tolist()))

    print("Generated sequence:", [len(seq) for seq in generated_seq])
