"""
MLflow Artifact Callbacks
--------------------------
MLflowCheckpointCallback      – Loggt den besten Checkpoint als MLflow-Artefakt.
SequencePredictionPlotCallback – Loggt Diagnose-Plots alle N Validierungs-Epochen
                                  (mit Epoch-Nummer im Dateinamen).
BestModelPlotCallback          – Loggt Plots für das aktuell beste Modell;
                                  Dateien werden immer überschrieben wenn ein
                                  besseres Ergebnis erzielt wurde.

Artefakt-Struktur (cadtoseq):
  checkpoints/        – Beste Modell-Checkpoints
  plots/examples/     – Vorhersage-Tabelle (epochenweise)
  plots/confusion/    – Token-Konfusionsmatrix (epochenweise)
  plots/levenshtein/  – Levenshtein-Distanzverteilung (epochenweise)
  plots/token_acc/    – Token-wise Accuracy (epochenweise)
  plots/best/         – Alle vier Plots für das aktuell beste Modell (wird überschrieben)
"""

import logging
import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from mpp.constants import INV_VOCAB, VOCAB
from mpp.ml.metrics.sequences import Sequence_comparator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemeinsame Basis-Klasse mit Plot- und Prediction-Logik
# ---------------------------------------------------------------------------

class _SequencePlotMixin:
    """
    Mixin mit geteilter Prediction-Collection und Plot-Logik.
    Wird von SequencePredictionPlotCallback und BestModelPlotCallback genutzt.
    """

    n_examples: int
    _comparator: Sequence_comparator

    # ------------------------------------------------------------------
    # Predictions über den gesamten Val-Loader sammeln
    # ------------------------------------------------------------------

    def _collect_predictions(self, val_dl, pl_module):
        """Iteriert über den kompletten Val-Loader und gibt aggregierte Tensoren zurück."""
        all_tf_preds, all_gen_preds, all_targets = [], [], []

        pl_module.eval()
        with torch.no_grad():
            for vector_set, padded_targets in val_dl:
                vector_set = vector_set.to(pl_module.device)
                padded_targets = padded_targets.to(pl_module.device)

                logits = pl_module(vector_set, padded_targets[:, :-1])
                all_tf_preds.append(logits.argmax(dim=-1).cpu())
                all_targets.append(padded_targets[:, 1:].cpu())
                all_gen_preds.append(
                    pl_module.generate(vector_set, device=str(pl_module.device)).cpu()
                )

        return (
            torch.cat(all_tf_preds, dim=0),
            torch.cat(all_gen_preds, dim=0),
            torch.cat(all_targets, dim=0),
        )

    # ------------------------------------------------------------------
    # Alle vier Plots erzeugen und loggen
    # ------------------------------------------------------------------

    def _generate_plots(
        self, tf_preds, gen_preds, targets, title_prefix, run_id, artifact_dir, filename_prefix
    ):
        with tempfile.TemporaryDirectory() as tmp:
            self._plot_examples(tf_preds, gen_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix)
            self._plot_confusion_matrix(tf_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix)
            self._plot_levenshtein(gen_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix)
            self._plot_token_accuracy(tf_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix)

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    def _decode(self, token_ids):
        return [INV_VOCAB.get(int(t), "?") for t in token_ids if int(t) != VOCAB["PAD"]]

    def _log(self, fig, path, run_id, artifact_dir):
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        mlflow.MlflowClient().log_artifact(run_id, path, artifact_path=artifact_dir)

    # ------------------------------------------------------------------
    # Plot-Methoden
    # ------------------------------------------------------------------

    def _plot_examples(self, tf_preds, gen_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        n = min(self.n_examples, targets.size(0))
        fig, ax = plt.subplots(figsize=(15, n * 0.6 + 1.8))
        ax.axis("off")

        rows = []
        for i in range(n):
            valid_mask = targets[i] != VOCAB["PAD"]
            gt = " → ".join(self._decode(targets[i]))
            tf = " → ".join(self._decode(tf_preds[i][valid_mask]))
            ar = " → ".join(self._decode(gen_preds[i]))
            rows.append([gt, tf, ar])

        table = ax.table(
            cellText=rows,
            colLabels=["Ground Truth", "Teacher-Forced", "Autoregressive"],
            cellLoc="left",
            loc="center",
            colWidths=[0.35, 0.35, 0.30],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        for i, row in enumerate(rows):
            color = "#c8e6c9" if row[0] == row[2] else "#ffffff"
            for j in range(3):
                table[i + 1, j].set_facecolor(color)

        ax.set_title(
            f"{title_prefix} – Vorhersage-Beispiele  (grün = exakte AR-Übereinstimmung)",
            pad=12, fontsize=11,
        )
        path = os.path.join(tmp, f"{filename_prefix}examples.png")
        self._log(fig, path, run_id, f"{artifact_dir}/examples")

    def _plot_confusion_matrix(self, preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        mask = targets != VOCAB["PAD"]
        p = preds[mask].cpu().numpy()
        t = targets[mask].cpu().numpy()

        n = len(VOCAB)
        cm = np.zeros((n, n), dtype=int)
        for pi, ti in zip(p, t):
            cm[ti, pi] += 1

        labels = [INV_VOCAB[i] for i in range(n)]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Vorhergesagt")
        ax.set_ylabel("Ground Truth")
        ax.set_title(f"{title_prefix} – Token-Konfusionsmatrix (Teacher-Forced)")
        path = os.path.join(tmp, f"{filename_prefix}confusion.png")
        self._log(fig, path, run_id, f"{artifact_dir}/confusion")

    def _plot_levenshtein(self, gen_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        mask_t = self._comparator._create_mask(targets)
        mask_p = self._comparator._create_mask(gen_preds)
        dists = self._comparator.levenshtein_distance(
            gen_preds, targets, mask_p, mask_t
        ).cpu().numpy()

        max_dist = max(int(dists.max()), 1)

        # Vollständiger Bereich [0, max_dist] – fehlende Werte werden mit 0 aufgefüllt
        all_values = np.arange(0, max_dist + 1)
        counts = np.bincount(dists.astype(int), minlength=max_dist + 1)
        freq = counts / counts.sum()

        fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
        ax.plot(all_values, freq, linewidth=2, color="#4c72b0")
        ax.fill_between(all_values, freq, alpha=0.15, color="#4c72b0")
        ax.axvline(dists.mean(), color="red", linestyle="--",
                   label=f"Mittelwert: {dists.mean():.2f}")
        ax.set_xticks(all_values)
        ax.set_xlabel("Levenshtein-Distanz")
        ax.set_ylabel("Relative Häufigkeit")
        ax.set_title(f"{title_prefix} – Levenshtein-Distanz (Autoregressive)")
        ax.legend()
        path = os.path.join(tmp, f"{filename_prefix}levenshtein.png")
        self._log(fig, path, run_id, f"{artifact_dir}/levenshtein")

    def _plot_token_accuracy(self, preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        token_acc = self._comparator.stepwise_accuracy(preds, targets)
        sorted_keys = sorted(token_acc.keys())
        labels = [INV_VOCAB[k] for k in sorted_keys]
        values = [token_acc[k] if not np.isnan(token_acc[k]) else 0.0
                  for k in sorted_keys]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(labels, values, edgecolor="black", color="#55a868")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{title_prefix} – Token-wise Accuracy (Teacher-Forced)")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        path = os.path.join(tmp, f"{filename_prefix}token_acc.png")
        self._log(fig, path, run_id, f"{artifact_dir}/token_acc")


# ---------------------------------------------------------------------------
# Checkpoint-Callback
# ---------------------------------------------------------------------------

class MLflowCheckpointCallback(Callback):
    """
    Loggt den besten Checkpoint als MLflow-Artefakt unter 'checkpoints/',
    sobald ModelCheckpoint eine neue beste Version speichert.
    """

    def __init__(self):
        self._last_logged: str = ""

    def on_validation_epoch_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, MLFlowLogger):
            return
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                if cb.best_model_path != self._last_logged:
                    mlflow.MlflowClient().log_artifact(
                        trainer.logger.run_id,
                        cb.best_model_path,
                        artifact_path="checkpoints",
                    )
                    self._last_logged = cb.best_model_path
                    logger.info(f"Checkpoint nach MLflow geloggt: {cb.best_model_path}")


# ---------------------------------------------------------------------------
# Epochenweise Plots
# ---------------------------------------------------------------------------

class SequencePredictionPlotCallback(_SequencePlotMixin, Callback):
    """
    Loggt vier Diagnose-Plots alle `plot_every_n_epochs` Epochen als MLflow-Artefakte.
    Dateinamen enthalten die Epochennummer → Verlauf bleibt vollständig erhalten.

    Parameters
    ----------
    plot_every_n_epochs : int
        Frequenz der Plot-Erzeugung.
    n_examples : int
        Anzahl der Beispiele in der Vorhersage-Tabelle.
    """

    def __init__(self, plot_every_n_epochs: int = 10, n_examples: int = 8):
        self.plot_every_n_epochs = plot_every_n_epochs
        self.n_examples = n_examples
        self._comparator = Sequence_comparator(VOCAB)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.plot_every_n_epochs != 0:
            return
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        val_dl = trainer.val_dataloaders
        if isinstance(val_dl, list):
            val_dl = val_dl[0]

        tf_preds, gen_preds, targets = self._collect_predictions(val_dl, pl_module)

        epoch = trainer.current_epoch
        self._generate_plots(
            tf_preds, gen_preds, targets,
            title_prefix=f"Epoch {epoch}",
            run_id=trainer.logger.run_id,
            artifact_dir="plots",
            filename_prefix=f"ep{epoch:04d}_",
        )


# ---------------------------------------------------------------------------
# Best-Model Plots (werden überschrieben)
# ---------------------------------------------------------------------------

class BestModelPlotCallback(_SequencePlotMixin, Callback):
    """
    Loggt vier Diagnose-Plots für das aktuell beste Modell unter 'plots/best/'.
    Die Dateien haben feste Namen und werden überschrieben, sobald ein
    besseres Ergebnis erzielt wird.

    Parameters
    ----------
    n_examples : int
        Anzahl der Beispiele in der Vorhersage-Tabelle.
    """

    def __init__(self, n_examples: int = 8):
        self.n_examples = n_examples
        self._comparator = Sequence_comparator(VOCAB)
        self._last_best_path: str = ""

    def on_validation_epoch_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        # Nur auslösen wenn ModelCheckpoint ein neues Bestes gespeichert hat
        best_path = ""
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                best_path = cb.best_model_path
                break

        if not best_path or best_path == self._last_best_path:
            return

        self._last_best_path = best_path

        val_dl = trainer.val_dataloaders
        if isinstance(val_dl, list):
            val_dl = val_dl[0]

        tf_preds, gen_preds, targets = self._collect_predictions(val_dl, pl_module)

        epoch = trainer.current_epoch
        logger.info(f"Neues bestes Modell (Epoch {epoch}) – Best-Plots werden überschrieben.")

        self._generate_plots(
            tf_preds, gen_preds, targets,
            title_prefix=f"Bestes Modell – Epoch {epoch}",
            run_id=trainer.logger.run_id,
            artifact_dir="plots/best",
            filename_prefix="",
        )
