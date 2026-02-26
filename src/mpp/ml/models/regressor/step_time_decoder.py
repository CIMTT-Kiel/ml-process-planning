"""Decoder-Architektur für die schrittweise Fertigungszeitvorhersage.

Dieses Modul enthält zwei Klassen:

* :class:`StepTimeDecoder` – Der eigentliche Transformer-Decoder, der die
  Geometrie-Repräsentation (Encoder-Memory) mit der Schrittsequenz verknüpft
  und pro Schritt eine skalare Zeit vorhersagt.

* :class:`TrsfmEncoderStepTimeModel` – Kombiniert einen vortrainierten
  :class:`~mpp.ml.models.regressor.trsfm_encoder_regressor.TrsfmEncoderRegressor`
  (Encoder) mit dem neuen Decoder zu einem Gesamt-Modell.  Unterstützt sowohl
  Teacher-Forcing (Training) als auch autoregressive Dekodierung (Inferenz).

Typische Verwendung
-------------------
>>> encoder = TrsfmEncoderRegressor(embed_dim=128)
>>> decoder = StepTimeDecoder(embed_dim=128)
>>> model   = TrsfmEncoderStepTimeModel(encoder, decoder)
>>> # Teacher Forcing:
>>> preds = model(vecset, step_tokens, prev_times)  # [B, seq_len]
>>> # Autoregressive Inferenz:
>>> preds = model.generate(vecset, step_tokens)      # [B, seq_len]
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from mpp.constants import VOCAB
from mpp.ml.models.regressor.trsfm_encoder_regressor import TrsfmEncoderRegressor

logger = logging.getLogger(__name__)


class StepTimeDecoder(nn.Module):
    """Transformer-Decoder für schrittweise Zeitvorhersage.

    Nimmt die vollständige Sequenz von Fertigungsschritten als Learned-Token-
    Embeddings entgegen und schätzt für jeden Schritt eine Dauer in Minuten.
    Dabei wird die (normalisierte) Zeit des jeweils vorangegangenen Schritts
    als zusätzliches skalares Feature auf die Embedding-Dimension projiziert
    und zu den Schritt-Embeddings addiert.

    Die Cross-Attention läuft gegen die Encoder-Ausgabe (Memory) des Geometrie-
    Encoders, so dass die Schätzung geometrische Information einbeziehen kann.

    Parameters
    ----------
    vocab_size : int
        Größe des Token-Vokabulars (inkl. PAD, START, STOP).
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

        # Gelernte Token-Embeddings (PAD wird auf Null initialisiert und nicht
        # geupdatet, weil er durch das Causal-Masking sowieso maskiert wird)
        self.step_embeddings = nn.Embedding(
            vocab_size, embed_dim, padding_idx=VOCAB["PAD"]
        )

        # NOTE: Gelernte Positional Embeddings (keine sinusoidalen), wie in der
        # Spezifikation gefordert.  Längere Sequenzen als max_seq_len werden
        # zur Laufzeit abgeschnitten.
        self.pos_embeddings = nn.Embedding(max_seq_len, embed_dim)

        # Projektion: skalare Vorschritt-Zeit → Embedding-Dim
        self.time_proj = nn.Linear(1, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Ausgabe-Projektion: Embedding-Dim → skalare Zeitschätzung
        self.output_proj = nn.Linear(embed_dim, 1)

        self.max_seq_len = max_seq_len

    def forward(
        self,
        memory: torch.Tensor,
        step_tokens: torch.Tensor,
        prev_times: torch.Tensor,
    ) -> torch.Tensor:
        """Parallelisierter Forward-Pass (Teacher Forcing oder Inferenz).

        Parameters
        ----------
        memory : torch.Tensor
            Encoder-Ausgabe der Form ``[B, set_size, embed_dim]``.
        step_tokens : torch.Tensor
            Token-IDs der Fertigungsschritte, Form ``[B, seq_len]``.
            PAD-Tokens werden automatisch maskiert.
        prev_times : torch.Tensor
            Normalisierte Zeiten des *vorangegangenen* Schritts,
            Form ``[B, seq_len]``.
            ``prev_times[:, 0] == 0.0`` (kein Vorgänger für Schritt 0).
            Im Teacher-Forcing-Modus: GT-Zeiten, um eine Position nach rechts
            verschoben.  Im autoregressiven Modus: eigene Vorhersagen.

        Returns
        -------
        torch.Tensor
            Vorhergesagte (normalisierte) Zeiten, Form ``[B, seq_len]``.
        """
        B, seq_len = step_tokens.shape

        # --- Token- und Positions-Embeddings ---
        step_emb = self.step_embeddings(step_tokens)           # [B, seq_len, E]
        positions = torch.arange(seq_len, device=step_tokens.device)
        pos_emb = self.pos_embeddings(positions)               # [seq_len, E]
        tgt = step_emb + pos_emb.unsqueeze(0)                  # [B, seq_len, E]

        # --- Vorschritt-Zeit einbetten und addieren ---
        time_emb = self.time_proj(prev_times.unsqueeze(-1))    # [B, seq_len, E]
        tgt = tgt + time_emb

        # --- Causal Mask (kein Look-ahead) ---
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=step_tokens.device
        ).bool()

        # --- Key-Padding-Mask für PAD-Tokens ---
        tgt_key_padding_mask = step_tokens == VOCAB["PAD"]     # [B, seq_len]

        # --- Transformer-Decoder ---
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # --- Skalarprojektion ---
        pred_times = self.output_proj(output).squeeze(-1)      # [B, seq_len]
        return pred_times


class TrsfmEncoderStepTimeModel(nn.Module):
    """Gesamtmodell: Geometrie-Encoder + Schrittzeit-Decoder.

    Der Encoder übernimmt Gewichte aus einem vortrainierten
    :class:`~mpp.ml.models.regressor.trsfm_encoder_regressor.TrsfmEncoderRegressor`
    und kann initial eingefroren werden.  Der Decoder ist neu initialisiert.

    Parameters
    ----------
    encoder : TrsfmEncoderRegressor
        Vortrainierter (oder frisch initialisierter) Geometrie-Encoder.
    decoder : StepTimeDecoder
        Neu initialisierter Schrittzeit-Decoder.
    """

    def __init__(
        self,
        encoder: TrsfmEncoderRegressor,
        decoder: StepTimeDecoder,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        vecset: torch.Tensor,
        step_tokens: torch.Tensor,
        prev_times: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-Forcing-Forward (Training).

        Parameters
        ----------
        vecset : torch.Tensor
            Geometrie-Embeddings ``[B, 1024, 32]``.
        step_tokens : torch.Tensor
            Fertigungsschritt-Token-IDs ``[B, seq_len]``.
        prev_times : torch.Tensor
            Normalisierte GT-Zeiten, um eine Position nach rechts verschoben
            ``[B, seq_len]``.

        Returns
        -------
        torch.Tensor
            Vorhergesagte Zeiten (normalisiert) ``[B, seq_len]``.
        """
        memory = self.encoder.encode(vecset)                   # [B, set_size, E]
        return self.decoder(memory, step_tokens, prev_times)

    @torch.no_grad()
    def generate_stream(
        self,
        vecset: torch.Tensor,
        step_tokens: torch.Tensor,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        """Autoregressive Inferenz als Generator – liefert pro Schritt ein Ergebnis.

        Anstatt den gesamten Tensor auf einmal zurückzugeben, wird nach jedem
        vorhergesagten Schritt ein Tupel geliefert.  Das ermöglicht z. B. in
        Streamlit eine schrittweise Anzeige, während die Berechnung noch läuft.

        Parameters
        ----------
        vecset : torch.Tensor
            Geometrie-Embeddings ``[1, 1024, 32]`` (Batch-Größe 1).
        step_tokens : torch.Tensor
            Fertigungsschritt-Token-IDs ``[1, seq_len]``.
        target_mean : float
            Mittelwert der Schrittzeiten aus dem Training (für Denormalisierung).
        target_std : float
            Standardabweichung der Schrittzeiten aus dem Training.

        Yields
        ------
        tuple[int, int, float]
            ``(schritt_index, token_id, vorhergesagte_zeit_in_minuten)``
            Nur für nicht-PAD-Positionen; stoppt beim ersten PAD-Token.
        """
        B, seq_len = step_tokens.shape
        memory = self.encoder.encode(vecset)                   # [1, set_size, E]
        all_pred_norm = torch.zeros(B, seq_len, device=vecset.device)

        for i in range(seq_len):
            token_id = int(step_tokens[0, i].item())
            if token_id == VOCAB["PAD"]:
                return

            prev_so_far = torch.zeros(B, i + 1, device=vecset.device)
            if i > 0:
                prev_so_far[:, 1:] = all_pred_norm[:, :i]

            tokens_so_far = step_tokens[:, : i + 1]
            out = self.decoder(memory, tokens_so_far, prev_so_far)
            t_norm = out[0, -1].item()                         # normiert
            all_pred_norm[0, i] = t_norm

            t_abs = t_norm * target_std + target_mean          # in Minuten
            yield i, token_id, t_abs

    @torch.no_grad()
    def generate(
        self,
        vecset: torch.Tensor,
        step_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive Inferenz: jede vorhergesagte Zeit wird als Eingabe
        für den nächsten Schritt verwendet.

        NOTE: Für kurze Sequenzen (max. ~8 Schritte) ist der O(n²)-Aufwand
        durch Neuberechnung des Decoders in jedem Schritt vernachlässigbar.
        Für längere Sequenzen kann ein KV-Cache ergänzt werden.

        Parameters
        ----------
        vecset : torch.Tensor
            Geometrie-Embeddings ``[B, 1024, 32]``.
        step_tokens : torch.Tensor
            Fertigungsschritt-Token-IDs ``[B, seq_len]``.  PAD-Tokens werden
            mit Zeitschätzung 0.0 zurückgegeben.

        Returns
        -------
        torch.Tensor
            Vorhergesagte Zeiten (normalisiert) ``[B, seq_len]``.
        """
        B, seq_len = step_tokens.shape
        memory = self.encoder.encode(vecset)                   # [B, set_size, E]

        all_pred_times = torch.zeros(B, seq_len, device=vecset.device)

        for i in range(seq_len):
            # Bau prev_times für Positionen 0..i:
            # Position 0 → 0.0, Position k → pred_{k-1}
            prev_so_far = torch.zeros(B, i + 1, device=vecset.device)
            if i > 0:
                prev_so_far[:, 1:] = all_pred_times[:, :i]

            tokens_so_far = step_tokens[:, : i + 1]           # [B, i+1]
            out = self.decoder(memory, tokens_so_far, prev_so_far)  # [B, i+1]
            t_i = out[:, -1]                                   # [B]

            # PAD-Positionen auf 0.0 setzen
            is_pad = tokens_so_far[:, -1] == VOCAB["PAD"]
            t_i = t_i.masked_fill(is_pad, 0.0)

            all_pred_times[:, i] = t_i

        return all_pred_times                                  # [B, seq_len]


if __name__ == "__main__":
    # Schnell-Smoke-Test
    B, SET_SIZE, INPUT_DIM = 4, 1024, 32
    SEQ_LEN, EMBED_DIM = 5, 128

    encoder = TrsfmEncoderRegressor(input_dim=INPUT_DIM, embed_dim=EMBED_DIM, num_heads=8, num_layers=2)
    decoder = StepTimeDecoder(embed_dim=EMBED_DIM, num_heads=8, num_layers=2)
    model = TrsfmEncoderStepTimeModel(encoder, decoder)

    vecset = torch.randn(B, SET_SIZE, INPUT_DIM)
    tokens = torch.randint(0, 7, (B, SEQ_LEN))
    prev_t = torch.randn(B, SEQ_LEN)

    # Teacher Forcing
    out_tf = model(vecset, tokens, prev_t)
    print(f"Teacher-Forcing output shape: {out_tf.shape}")   # [4, 5]

    # Autoregressive
    out_ar = model.generate(vecset, tokens)
    print(f"Autoregressive output shape:  {out_ar.shape}")   # [4, 5]
