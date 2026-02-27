"""Decoder-Architektur für die MTL-Schritt-Zeit-und-Kosten-Vorhersage.

Dieses Modul enthält zwei Klassen:

* :class:`MTLStepTimeDecoder` – Transformer-Decoder der für jeden Schritt
  gleichzeitig eine Zeit *und* Kosten vorhersagt.  Die Vorschritt-Zeit und
  -Kosten werden beide als skalare Features eingebettet und addiert.

* :class:`MTLEncoderDecoderModel` – Kombiniert einen
  :class:`~mpp.ml.models.regressor.trsfm_encoder_regressor.TrsfmEncoderRegressor`
  (Encoder) mit dem MTL-Decoder.  Unterstützt Teacher-Forcing (Training) und
  autoregressive Dekodierung (Inferenz).

Typische Verwendung
-------------------
>>> encoder = TrsfmEncoderRegressor(embed_dim=128)
>>> decoder = MTLStepTimeDecoder(embed_dim=128)
>>> model   = MTLEncoderDecoderModel(encoder, decoder)
>>> # Teacher Forcing:
>>> pred_t, pred_c = model(vecset, step_tokens, prev_times, prev_costs)
>>> # Autoregressive Inferenz:
>>> pred_t, pred_c = model.generate(vecset, step_tokens)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from mpp.constants import VOCAB, INV_VOCAB
from mpp.ml.models.regressor.trsfm_encoder_regressor import TrsfmEncoderRegressor

logger = logging.getLogger(__name__)


class MTLStepTimeDecoder(nn.Module):
    """Transformer-Decoder für simultane Schritt-Zeit- und -Kosten-Vorhersage.

    Identische Grundstruktur wie ``StepTimeDecoder``, erweitert um eine zweite
    Eingabe-Projektion (``cost_proj``) und einen zweiten Ausgabe-Kopf
    (``cost_head``).  Encoder und Decoder teilen sich einen ``nn.TransformerDecoder``.

    Parameters
    ----------
    vocab_size : int
        Größe des Token-Vokabulars.
    embed_dim : int
        Embedding-Dimension (muss mit dem Encoder übereinstimmen).
    num_heads : int
        Anzahl der Attention-Köpfe.
    num_layers : int
        Anzahl der Transformer-Decoder-Layer.
    dropout : float
        Dropout-Rate.
    max_seq_len : int
        Maximale Sequenzlänge (für gelernte Positional-Embeddings).
    """

    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 12,
    ) -> None:
        super().__init__()

        self.step_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=VOCAB["PAD"])
        self.pos_embeddings  = nn.Embedding(max_seq_len, embed_dim)

        # Zwei skalare Vorschritt-Features → Embedding-Dim
        self.time_proj = nn.Linear(1, embed_dim)
        self.cost_proj = nn.Linear(1, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Zwei unabhängige Ausgabeköpfe
        self.time_head = nn.Linear(embed_dim, 1)
        self.cost_head = nn.Linear(embed_dim, 1)

        self.max_seq_len = max_seq_len

    def forward(
        self,
        memory: torch.Tensor,
        step_tokens: torch.Tensor,
        prev_times: torch.Tensor,
        prev_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parallelisierter Forward-Pass (Teacher Forcing).

        Parameters
        ----------
        memory : torch.Tensor
            Encoder-Ausgabe ``[B, set_size, embed_dim]``.
        step_tokens : torch.Tensor
            Token-IDs ``[B, seq_len]``.
        prev_times : torch.Tensor
            Normalisierte Vorschritt-Zeiten ``[B, seq_len]``.
            ``prev_times[:, 0] == 0.0`` (kein Vorgänger für Schritt 0).
        prev_costs : torch.Tensor
            Normalisierte Vorschritt-Kosten ``[B, seq_len]``.
            ``prev_costs[:, 0] == 0.0`` (kein Vorgänger für Schritt 0).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - ``pred_times`` : normalisierte Zeitvorhersagen ``[B, seq_len]``
            - ``pred_costs`` : normalisierte Kostenvorhersagen ``[B, seq_len]``
        """
        B, seq_len = step_tokens.shape

        step_emb = self.step_embeddings(step_tokens)              # [B, seq_len, E]
        positions = torch.arange(seq_len, device=step_tokens.device)
        pos_emb = self.pos_embeddings(positions)                  # [seq_len, E]
        tgt = step_emb + pos_emb.unsqueeze(0)                     # [B, seq_len, E]

        # Vorschritt-Features einbetten und addieren
        time_emb = self.time_proj(prev_times.unsqueeze(-1))       # [B, seq_len, E]
        cost_emb = self.cost_proj(prev_costs.unsqueeze(-1))       # [B, seq_len, E]
        tgt = tgt + time_emb + cost_emb

        # Causal Mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=step_tokens.device
        ).bool()

        tgt_key_padding_mask = step_tokens == VOCAB["PAD"]        # [B, seq_len]

        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )                                                          # [B, seq_len, E]

        pred_times = self.time_head(output).squeeze(-1)            # [B, seq_len]
        pred_costs = self.cost_head(output).squeeze(-1)            # [B, seq_len]
        return pred_times, pred_costs


class MTLEncoderDecoderModel(nn.Module):
    """Gesamtmodell: Geometrie-Encoder + MTL-Schritt-Decoder.

    Parameters
    ----------
    encoder : TrsfmEncoderRegressor
        Vortrainierter (oder frisch initialisierter) Geometrie-Encoder.
    decoder : MTLStepTimeDecoder
        MTL-Decoder mit Zeit- und Kostenkopf.
    """

    def __init__(
        self,
        encoder: TrsfmEncoderRegressor,
        decoder: MTLStepTimeDecoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        vecset: torch.Tensor,
        step_tokens: torch.Tensor,
        prev_times: torch.Tensor,
        prev_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-Forcing-Forward.

        Parameters
        ----------
        vecset : torch.Tensor
            Geometrie-Embeddings ``[B, 1024, 32]``.
        step_tokens : torch.Tensor
            Token-IDs ``[B, seq_len]``.
        prev_times : torch.Tensor
            Normalisierte GT-Zeiten, um 1 nach rechts verschoben ``[B, seq_len]``.
        prev_costs : torch.Tensor
            Normalisierte GT-Kosten, um 1 nach rechts verschoben ``[B, seq_len]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(pred_times_norm, pred_costs_norm)`` jeweils ``[B, seq_len]``.
        """
        memory = self.encoder.encode(vecset)                       # [B, set_size, E]
        return self.decoder(memory, step_tokens, prev_times, prev_costs)

    @torch.no_grad()
    def generate(
        self,
        vecset: torch.Tensor,
        step_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive Inferenz – liefert alle Zeiten und Kosten auf einmal.

        Jede vorhergesagte Zeit/Kosten wird als Eingabe für den nächsten Schritt
        verwendet.

        Parameters
        ----------
        vecset : torch.Tensor
            Geometrie-Embeddings ``[B, 1024, 32]``.
        step_tokens : torch.Tensor
            Token-IDs ``[B, seq_len]``.  PAD-Positionen erhalten 0.0.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(pred_times_norm, pred_costs_norm)`` jeweils ``[B, seq_len]``,
            normalisiert.
        """
        B, seq_len = step_tokens.shape
        memory = self.encoder.encode(vecset)                       # [B, set_size, E]

        all_pred_t = torch.zeros(B, seq_len, device=vecset.device)
        all_pred_c = torch.zeros(B, seq_len, device=vecset.device)

        for i in range(seq_len):
            prev_t_so_far = torch.zeros(B, i + 1, device=vecset.device)
            prev_c_so_far = torch.zeros(B, i + 1, device=vecset.device)
            if i > 0:
                prev_t_so_far[:, 1:] = all_pred_t[:, :i]
                prev_c_so_far[:, 1:] = all_pred_c[:, :i]

            tokens_so_far = step_tokens[:, : i + 1]               # [B, i+1]
            out_t, out_c = self.decoder(
                memory, tokens_so_far, prev_t_so_far, prev_c_so_far
            )                                                      # [B, i+1]

            t_i = out_t[:, -1]                                     # [B]
            c_i = out_c[:, -1]                                     # [B]

            is_pad = tokens_so_far[:, -1] == VOCAB["PAD"]
            t_i = t_i.masked_fill(is_pad, 0.0)
            c_i = c_i.masked_fill(is_pad, 0.0)

            all_pred_t[:, i] = t_i
            all_pred_c[:, i] = c_i

        return all_pred_t, all_pred_c

    @torch.no_grad()
    def generate_stream(
        self,
        vecset: torch.Tensor,
        step_tokens: torch.Tensor,
        target_mean_time: float = 0.0,
        target_std_time: float = 1.0,
        target_mean_cost: float = 0.0,
        target_std_cost: float = 1.0,
    ):
        """Autoregressive Inferenz als Generator – liefert pro Schritt ein Ergebnis.

        Parameters
        ----------
        vecset : torch.Tensor
            Geometrie-Embeddings ``[1, 1024, 32]`` (Batch-Größe 1).
        step_tokens : torch.Tensor
            Token-IDs ``[1, seq_len]``.
        target_mean_time, target_std_time : float
            Normalisierungsparameter für die Zeit.
        target_mean_cost, target_std_cost : float
            Normalisierungsparameter für die Kosten.

        Yields
        ------
        tuple[int, int, float, float]
            ``(schritt_index, token_id, zeit_in_minuten, kosten_in_dollar)``
            Nur für nicht-PAD-Positionen; stoppt beim ersten PAD-Token.
        """
        B, seq_len = step_tokens.shape
        memory = self.encoder.encode(vecset)                       # [1, set_size, E]

        all_pred_t = torch.zeros(B, seq_len, device=vecset.device)
        all_pred_c = torch.zeros(B, seq_len, device=vecset.device)

        for i in range(seq_len):
            token_id = int(step_tokens[0, i].item())
            if token_id == VOCAB["PAD"]:
                return

            prev_t_so_far = torch.zeros(B, i + 1, device=vecset.device)
            prev_c_so_far = torch.zeros(B, i + 1, device=vecset.device)
            if i > 0:
                prev_t_so_far[:, 1:] = all_pred_t[:, :i]
                prev_c_so_far[:, 1:] = all_pred_c[:, :i]

            tokens_so_far = step_tokens[:, : i + 1]
            out_t, out_c = self.decoder(memory, tokens_so_far, prev_t_so_far, prev_c_so_far)

            t_norm = out_t[0, -1].item()
            c_norm = out_c[0, -1].item()
            all_pred_t[0, i] = t_norm
            all_pred_c[0, i] = c_norm

            t_abs = t_norm * target_std_time + target_mean_time
            c_abs = c_norm * target_std_cost + target_mean_cost
            yield i, token_id, t_abs, c_abs


if __name__ == "__main__":
    # Schnell-Smoke-Test
    B, SET_SIZE, INPUT_DIM = 4, 1024, 32
    SEQ_LEN, EMBED_DIM = 5, 128

    encoder = TrsfmEncoderRegressor(input_dim=INPUT_DIM, embed_dim=EMBED_DIM, num_heads=8, num_layers=2)
    decoder = MTLStepTimeDecoder(embed_dim=EMBED_DIM, num_heads=8, num_layers=2)
    model   = MTLEncoderDecoderModel(encoder, decoder)

    vecset      = torch.randn(B, SET_SIZE, INPUT_DIM)
    tokens      = torch.randint(0, 7, (B, SEQ_LEN))
    prev_t      = torch.randn(B, SEQ_LEN)
    prev_c      = torch.randn(B, SEQ_LEN)

    # Teacher Forcing
    out_t, out_c = model(vecset, tokens, prev_t, prev_c)
    print(f"Teacher-Forcing time shape: {out_t.shape}")    # [4, 5]
    print(f"Teacher-Forcing cost shape: {out_c.shape}")    # [4, 5]

    # Autoregressive
    ar_t, ar_c = model.generate(vecset, tokens)
    print(f"Autoregressive time shape:  {ar_t.shape}")     # [4, 5]
    print(f"Autoregressive cost shape:  {ar_c.shape}")     # [4, 5]
