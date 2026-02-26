"""
MLflow Artifact Callbacks
--------------------------
MLflowCheckpointCallback          – Loggt den besten Checkpoint als MLflow-Artefakt.
SequencePredictionPlotCallback     – Loggt Diagnose-Plots alle N Validierungs-Epochen
                                     (mit Epoch-Nummer im Dateinamen).
BestModelPlotCallback              – Loggt Plots für das aktuell beste Modell;
                                     Dateien werden immer überschrieben wenn ein
                                     besseres Ergebnis erzielt wurde.
RegressionPredictionPlotCallback   – Loggt Regressions-Diagnose-Plots alle N Epochen.
BestRegressionModelPlotCallback    – Loggt Regressions-Plots für das aktuell beste Modell.
StepTimePredictionPlotCallback     – Loggt Schrittzeit-Diagnose-Plots alle N Epochen.
BestStepTimeModelPlotCallback      – Loggt Schrittzeit-Plots für das aktuell beste Modell.

Artefakt-Struktur (cadtoseq):
  checkpoints/        – Beste Modell-Checkpoints
  plots/examples/     – Vorhersage-Tabelle (epochenweise)
  plots/confusion/    – Token-Konfusionsmatrix (epochenweise)
  plots/levenshtein/  – Levenshtein-Distanzverteilung (epochenweise)
  plots/token_acc/    – Token-wise Accuracy (epochenweise)
  plots/best/         – Alle vier Plots für das aktuell beste Modell (wird überschrieben)

Artefakt-Struktur (process-time-regression):
  checkpoints/        – Beste Modell-Checkpoints
  plots/scatter/      – Vorhergesagt vs. tatsächlich (epochenweise)
  plots/residuals/    – Residuen-Plot (epochenweise)
  plots/errors/       – Fehlerverteilung (epochenweise)
  plots/best/         – Alle drei Plots für das aktuell beste Modell (wird überschrieben)

Artefakt-Struktur (cadtostepset):
  checkpoints/        – Beste Modell-Checkpoints
  plots/examples/     – Vorhersage-Tabelle (epochenweise)
  plots/class_metrics/– Precision/Recall pro Prozessschritt (epochenweise)
  plots/best/         – Beide Plots für das aktuell beste Modell (wird überschrieben)

Artefakt-Struktur (step-time-regression):
  checkpoints/        – Beste Modell-Checkpoints
  plots/scatter/      – Schrittzeit Vorhergesagt vs. Tatsächlich, nach Token eingefärbt
  plots/token_mae/    – MAE pro Prozessschritttyp (epochenweise)
  plots/consistency/  – Σ Schrittzeiten vs. GT-Gesamtzeit (epochenweise)
  plots/errors/       – Fehlerverteilung Schrittzeiten (epochenweise)
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
import torch.nn.functional as F
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from mpp.constants import INV_VOCAB, VOCAB
from mpp.ml.metrics.sequences import Sequence_comparator

logger = logging.getLogger(__name__)

# Prozessschritt-Labels (ohne START, STOP, PAD), in Reihenfolge der Token-IDs
_SPECIAL_TOKENS = {"START", "STOP", "PAD"}
STEP_LABELS = [INV_VOCAB[i] for i in range(len(VOCAB)) if INV_VOCAB[i] not in _SPECIAL_TOKENS]


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

        # Absolute Konfusionsmatrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Vorhergesagt")
        ax.set_ylabel("Ground Truth")
        ax.set_title(f"{title_prefix} – Token-Konfusionsmatrix absolut (Teacher-Forced)")
        path = os.path.join(tmp, f"{filename_prefix}confusion.png")
        self._log(fig, path, run_id, f"{artifact_dir}/confusion")

        # Relative Konfusionsmatrix (zeilenweise normiert, d. h. Recall pro Klasse)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_rel = np.where(row_sums > 0, cm / row_sums, 0.0)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_rel, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0,
                    xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_xlabel("Vorhergesagt")
        ax2.set_ylabel("Ground Truth")
        ax2.set_title(f"{title_prefix} – Token-Konfusionsmatrix relativ (Teacher-Forced)")
        path2 = os.path.join(tmp, f"{filename_prefix}confusion_rel.png")
        self._log(fig2, path2, run_id, f"{artifact_dir}/confusion")

    def _plot_levenshtein(self, gen_preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        mask_t = self._comparator._create_mask(targets)
        mask_p = self._comparator._create_mask(gen_preds)
        dists = self._comparator.levenshtein_distance(
            gen_preds, targets, mask_p, mask_t
        ).cpu().numpy().astype(float)

        fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
        sns.kdeplot(data=dists, ax=ax, fill=True, color="#4c72b0",
                    alpha=0.15, linewidth=2, clip=(0.0, None), bw_adjust=0.4)
        ax.axvline(dists.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mittelwert: {dists.mean():.2f}")
        ax.set_xlabel("Levenshtein-Distanz")
        ax.set_ylabel("Dichte")
        ax.set_title(f"{title_prefix} – Levenshtein-Distanz (Autoregressive)")
        ax.legend()
        fig.tight_layout()
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
    Loggt die besten Checkpoints (save_top_k) als MLflow-Artefakte unter
    'checkpoints/' am Ende des Trainings. So werden genau die Top-k Modelle
    gespeichert – ohne Akkumulation aller Zwischenstände.
    """

    def on_train_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, MLFlowLogger):
            return
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                for path in cb.best_k_models:
                    if path:
                        mlflow.MlflowClient().log_artifact(
                            trainer.logger.run_id,
                            path,
                            artifact_path="checkpoints",
                        )
                        logger.info(f"Checkpoint nach MLflow geloggt: {path}")


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


# ---------------------------------------------------------------------------
# Regression: gemeinsame Basis-Klasse mit Plot-Logik
# ---------------------------------------------------------------------------

class _RegressionPlotMixin:
    """
    Mixin mit geteilter Prediction-Collection und Plot-Logik für Regressionsmodelle.
    Wird von RegressionPredictionPlotCallback und BestRegressionModelPlotCallback genutzt.
    """

    # ------------------------------------------------------------------
    # Predictions über den gesamten Val-Loader sammeln
    # ------------------------------------------------------------------

    def _collect_predictions(self, val_dl, pl_module):
        """Iteriert über den kompletten Val-Loader und gibt (preds, targets) als numpy-Arrays zurück.

        Vorhersagen werden denormalisiert (absolute Einheit der Zielgröße),
        sofern target_mean / target_std in den Hyperparametern hinterlegt sind.
        """
        all_preds, all_targets = [], []

        target_mean = getattr(pl_module.hparams, "target_mean", 0.0)
        target_std  = getattr(pl_module.hparams, "target_std",  1.0)

        pl_module.eval()
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(pl_module.device)
                preds_norm = pl_module(x).cpu()
                preds_abs  = preds_norm * target_std + target_mean
                all_preds.append(preds_abs)
                all_targets.append(y.cpu())

        preds_np   = torch.cat(all_preds).numpy()
        targets_np = torch.cat(all_targets).numpy()
        return preds_np, targets_np

    # ------------------------------------------------------------------
    # Alle drei Plots erzeugen und loggen
    # ------------------------------------------------------------------

    def _generate_plots(self, preds, targets, title_prefix, run_id, artifact_dir, filename_prefix):
        with tempfile.TemporaryDirectory() as tmp:
            self._plot_scatter(preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix)
            self._plot_residuals(preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix)
            self._plot_error_distribution(preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix)

    # ------------------------------------------------------------------
    # Hilfsmethod
    # ------------------------------------------------------------------

    def _log(self, fig, path, run_id, artifact_dir):
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        mlflow.MlflowClient().log_artifact(run_id, path, artifact_path=artifact_dir)

    # ------------------------------------------------------------------
    # Plot-Methoden
    # ------------------------------------------------------------------

    def _plot_scatter(self, preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(targets, preds, alpha=0.4, edgecolors="none", s=20, color="#4c72b0")
        min_val = min(targets.min(), preds.min())
        max_val = max(targets.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal")
        ax.set_xlabel("Tatsächlich [min]")
        ax.set_ylabel("Vorhergesagt [min]")
        ax.set_title(f"{title_prefix} – Vorhergesagt vs. Tatsächlich")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(tmp, f"{filename_prefix}scatter.png")
        self._log(fig, path, run_id, f"{artifact_dir}/scatter")

    def _plot_residuals(self, preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        residuals = preds - targets
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(targets, residuals, alpha=0.4, edgecolors="none", s=20, color="#55a868")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Tatsächlich [min]")
        ax.set_ylabel("Residuum [min] (Vorhergesagt – Tatsächlich)")
        ax.set_title(f"{title_prefix} – Residuen-Plot")
        fig.tight_layout()
        path = os.path.join(tmp, f"{filename_prefix}residuals.png")
        self._log(fig, path, run_id, f"{artifact_dir}/residuals")

    def _plot_error_distribution(self, preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        errors = preds - targets
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(errors, ax=ax, kde=True, color="#4c72b0", alpha=0.6)
        ax.axvline(errors.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mittelwert: {errors.mean():.2f} min")
        ax.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)
        ax.set_xlabel("Fehler [min] (Vorhergesagt – Tatsächlich)")
        ax.set_ylabel("Häufigkeit")
        ax.set_title(f"{title_prefix} – Fehlerverteilung")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(tmp, f"{filename_prefix}errors.png")
        self._log(fig, path, run_id, f"{artifact_dir}/errors")


# ---------------------------------------------------------------------------
# Epochenweise Regressions-Plots
# ---------------------------------------------------------------------------

class RegressionPredictionPlotCallback(_RegressionPlotMixin, Callback):
    """
    Loggt drei Regressions-Diagnose-Plots alle `plot_every_n_epochs` Epochen als MLflow-Artefakte.
    Dateinamen enthalten die Epochennummer → Verlauf bleibt vollständig erhalten.

    Parameters
    ----------
    plot_every_n_epochs : int
        Frequenz der Plot-Erzeugung.
    """

    def __init__(self, plot_every_n_epochs: int = 10):
        self.plot_every_n_epochs = plot_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.plot_every_n_epochs != 0:
            return
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        val_dl = trainer.val_dataloaders
        if isinstance(val_dl, list):
            val_dl = val_dl[0]

        preds, targets = self._collect_predictions(val_dl, pl_module)

        epoch = trainer.current_epoch
        self._generate_plots(
            preds, targets,
            title_prefix=f"Epoch {epoch}",
            run_id=trainer.logger.run_id,
            artifact_dir="plots",
            filename_prefix=f"ep{epoch:04d}_",
        )


# ---------------------------------------------------------------------------
# Best-Model Regressions-Plots (werden überschrieben)
# ---------------------------------------------------------------------------

class BestRegressionModelPlotCallback(_RegressionPlotMixin, Callback):
    """
    Loggt drei Regressions-Diagnose-Plots für das aktuell beste Modell unter 'plots/best/'.
    Die Dateien haben feste Namen und werden überschrieben, sobald ein besseres
    Ergebnis erzielt wird.
    """

    def __init__(self):
        self._last_best_path: str = ""

    def on_validation_epoch_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, MLFlowLogger):
            return

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

        preds, targets = self._collect_predictions(val_dl, pl_module)

        epoch = trainer.current_epoch
        logger.info(f"Neues bestes Modell (Epoch {epoch}) – Regressions-Best-Plots werden überschrieben.")

        self._generate_plots(
            preds, targets,
            title_prefix=f"Bestes Modell – Epoch {epoch}",
            run_id=trainer.logger.run_id,
            artifact_dir="plots/best",
            filename_prefix="",
        )


# ---------------------------------------------------------------------------
# Stepset: gemeinsame Basis-Klasse mit Plot-Logik
# ---------------------------------------------------------------------------

class _StepsetPlotMixin:
    """
    Mixin mit geteilter Prediction-Collection und Plot-Logik für Multi-Label-Klassifikation.
    Wird von StepsetPredictionPlotCallback und BestStepsetModelPlotCallback genutzt.
    """

    n_examples: int

    # ------------------------------------------------------------------
    # Predictions über den gesamten Val-Loader sammeln
    # ------------------------------------------------------------------

    def _collect_predictions(self, val_dl, pl_module):
        """Iteriert über den kompletten Val-Loader.

        Returns
        -------
        preds : torch.BoolTensor  [N, num_classes]
        targets : torch.BoolTensor  [N, num_classes]
        """
        all_preds, all_targets = [], []
        pl_module.eval()
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(pl_module.device)
                logits = pl_module(x).cpu()
                preds = torch.sigmoid(logits) > pl_module.hparams.threshold
                all_preds.append(preds)
                all_targets.append(y.bool().cpu())
        return torch.cat(all_preds), torch.cat(all_targets)

    # ------------------------------------------------------------------
    # Beide Plots erzeugen und loggen
    # ------------------------------------------------------------------

    def _generate_plots(self, preds, targets, title_prefix, run_id, artifact_dir, filename_prefix):
        with tempfile.TemporaryDirectory() as tmp:
            self._plot_examples(preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix)
            self._plot_class_metrics(preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix)

    # ------------------------------------------------------------------
    # Hilfsmethode
    # ------------------------------------------------------------------

    def _log(self, fig, path, run_id, artifact_dir):
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        mlflow.MlflowClient().log_artifact(run_id, path, artifact_path=artifact_dir)

    def _labels_from_mask(self, mask):
        """Wandelt einen bool-Vektor in eine lesbare Schritt-Liste um."""
        steps = [STEP_LABELS[i] for i, v in enumerate(mask) if v]
        return " | ".join(steps) if steps else "–"

    # ------------------------------------------------------------------
    # Plot-Methoden
    # ------------------------------------------------------------------

    def _plot_examples(self, preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        n = min(self.n_examples, targets.size(0))
        rows = []
        for i in range(n):
            gt = self._labels_from_mask(targets[i])
            pred = self._labels_from_mask(preds[i])
            rows.append([gt, pred])

        fig, ax = plt.subplots(figsize=(14, n * 0.65 + 1.8))
        ax.axis("off")
        table = ax.table(
            cellText=rows,
            colLabels=["Ground Truth", "Vorhergesagt"],
            cellLoc="left",
            loc="center",
            colWidths=[0.50, 0.50],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        for i, row in enumerate(rows):
            color = "#c8e6c9" if row[0] == row[1] else "#ffffff"
            for j in range(2):
                table[i + 1, j].set_facecolor(color)

        ax.set_title(
            f"{title_prefix} – Vorhersage-Beispiele  (grün = exakte Übereinstimmung)",
            pad=12, fontsize=11,
        )
        path = os.path.join(tmp, f"{filename_prefix}examples.png")
        self._log(fig, path, run_id, f"{artifact_dir}/examples")

    def _plot_class_metrics(self, preds, targets, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        n_classes = len(STEP_LABELS)

        precisions, recalls, f1s = [], [], []
        for c in range(n_classes):
            tp = ((preds_np[:, c] == 1) & (targets_np[:, c] == 1)).sum()
            fp = ((preds_np[:, c] == 1) & (targets_np[:, c] == 0)).sum()
            fn = ((preds_np[:, c] == 0) & (targets_np[:, c] == 1)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            precisions.append(float(p))
            recalls.append(float(r))
            f1s.append(float(f1))

        x = np.arange(n_classes)
        width = 0.25
        fig, ax = plt.subplots(figsize=(11, 5))
        bars_p = ax.bar(x - width, precisions, width, label="Precision", color="#4c72b0", edgecolor="black")
        bars_r = ax.bar(x,         recalls,    width, label="Recall",    color="#55a868", edgecolor="black")
        bars_f = ax.bar(x + width, f1s,        width, label="F1",        color="#c44e52", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(STEP_LABELS, rotation=20, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Wert")
        ax.set_title(f"{title_prefix} – Precision, Recall & F1 pro Prozessschritt")
        ax.legend()
        for bar, val in zip(list(bars_p) + list(bars_r) + list(bars_f), precisions + recalls + f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        path = os.path.join(tmp, f"{filename_prefix}class_metrics.png")
        self._log(fig, path, run_id, f"{artifact_dir}/class_metrics")


# ---------------------------------------------------------------------------
# Epochenweise Stepset-Plots
# ---------------------------------------------------------------------------

class StepsetPredictionPlotCallback(_StepsetPlotMixin, Callback):
    """
    Loggt zwei Diagnose-Plots alle `plot_every_n_epochs` Epochen als MLflow-Artefakte.
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

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.plot_every_n_epochs != 0:
            return
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        val_dl = trainer.val_dataloaders
        if isinstance(val_dl, list):
            val_dl = val_dl[0]

        preds, targets = self._collect_predictions(val_dl, pl_module)

        epoch = trainer.current_epoch
        self._generate_plots(
            preds, targets,
            title_prefix=f"Epoch {epoch}",
            run_id=trainer.logger.run_id,
            artifact_dir="plots",
            filename_prefix=f"ep{epoch:04d}_",
        )


# ---------------------------------------------------------------------------
# Best-Model Stepset-Plots (werden überschrieben)
# ---------------------------------------------------------------------------

class BestStepsetModelPlotCallback(_StepsetPlotMixin, Callback):
    """
    Loggt zwei Diagnose-Plots für das aktuell beste Modell unter 'plots/best/'.
    Die Dateien haben feste Namen und werden überschrieben, sobald ein besseres
    Ergebnis erzielt wird.

    Parameters
    ----------
    n_examples : int
        Anzahl der Beispiele in der Vorhersage-Tabelle.
    """

    def __init__(self, n_examples: int = 8):
        self.n_examples = n_examples
        self._last_best_path: str = ""

    def on_validation_epoch_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, MLFlowLogger):
            return

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

        preds, targets = self._collect_predictions(val_dl, pl_module)

        epoch = trainer.current_epoch
        logger.info(f"Neues bestes Modell (Epoch {epoch}) – Stepset-Best-Plots werden überschrieben.")

        self._generate_plots(
            preds, targets,
            title_prefix=f"Bestes Modell – Epoch {epoch}",
            run_id=trainer.logger.run_id,
            artifact_dir="plots/best",
            filename_prefix="",
        )


# ---------------------------------------------------------------------------
# StepTime-Regression: gemeinsame Basis-Klasse mit Plot-Logik
# ---------------------------------------------------------------------------

class _StepTimePlotMixin:
    """Mixin mit geteilter Prediction-Collection und Plot-Logik für die
    schrittweise Zeitvorhersage (step-time-regression).

    Der Val-DataLoader liefert Batches ``(vecset, step_tokens, step_times, total_time)``.
    Vorhersagen werden per Teacher Forcing erzeugt und denormalisiert.
    """

    # ------------------------------------------------------------------
    # Predictions über den gesamten Val-Loader sammeln
    # ------------------------------------------------------------------

    def _collect_predictions(self, val_dl, pl_module) -> dict:
        """Iteriert über den kompletten Val-Loader.

        Returns
        -------
        dict mit Feldern
            ``pred_flat``   – denormalisierte Vorhersagen je gültigem Schritt [N]
            ``gt_flat``     – GT-Schrittzeiten je gültigem Schritt [N]
            ``token_flat``  – Token-IDs je gültigem Schritt [N]
            ``pred_total``  – Summe der Vorhersagen je Sample [M]
            ``gt_total``    – GT-Gesamtzeit je Sample [M]
        """
        target_mean = getattr(pl_module.hparams, "target_mean", 0.0)
        target_std  = getattr(pl_module.hparams, "target_std",  1.0)

        all_pred_flat, all_gt_flat, all_tok_flat = [], [], []
        all_pred_total, all_gt_total = [], []

        pl_module.eval()
        with torch.no_grad():
            for vecset, step_tokens, step_times, total_time in val_dl:
                vecset      = vecset.to(pl_module.device)
                step_tokens = step_tokens.to(pl_module.device)
                step_times  = step_times.to(pl_module.device)
                total_time  = total_time.to(pl_module.device)

                # Teacher-Forcing-Forward
                step_times_norm = (step_times - target_mean) / target_std
                prev_times = F.pad(step_times_norm[:, :-1], (1, 0), value=0.0)
                pred_norm  = pl_module.model(vecset, step_tokens, prev_times)
                pred_abs   = pred_norm * target_std + target_mean

                pad_mask = step_tokens == VOCAB["PAD"]
                valid    = ~pad_mask

                all_pred_flat.append(pred_abs[valid].cpu())
                all_gt_flat.append(step_times[valid].cpu())
                all_tok_flat.append(step_tokens[valid].cpu())

                # Gesamtzeit pro Sample (PAD ausmaskieren)
                pred_total = pred_abs.masked_fill(pad_mask, 0.0).sum(dim=-1)
                all_pred_total.append(pred_total.cpu())
                all_gt_total.append(total_time.cpu())

        return {
            "pred_flat":  torch.cat(all_pred_flat).numpy(),
            "gt_flat":    torch.cat(all_gt_flat).numpy(),
            "token_flat": torch.cat(all_tok_flat).numpy(),
            "pred_total": torch.cat(all_pred_total).numpy(),
            "gt_total":   torch.cat(all_gt_total).numpy(),
        }

    # ------------------------------------------------------------------
    # Alle vier Plots erzeugen und loggen
    # ------------------------------------------------------------------

    def _generate_plots(self, data, title_prefix, run_id, artifact_dir, filename_prefix):
        with tempfile.TemporaryDirectory() as tmp:
            self._plot_scatter(data, title_prefix, tmp, run_id, artifact_dir, filename_prefix)
            self._plot_per_token_mae(data, title_prefix, tmp, run_id, artifact_dir, filename_prefix)
            self._plot_consistency(data, title_prefix, tmp, run_id, artifact_dir, filename_prefix)
            self._plot_error_distribution(data, title_prefix, tmp, run_id, artifact_dir, filename_prefix)

    # ------------------------------------------------------------------
    # Hilfsmethode
    # ------------------------------------------------------------------

    def _log(self, fig, path, run_id, artifact_dir):
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        mlflow.MlflowClient().log_artifact(run_id, path, artifact_path=artifact_dir)

    # ------------------------------------------------------------------
    # Plot-Methoden
    # ------------------------------------------------------------------

    def _plot_scatter(self, data, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        """Schrittzeit Vorhergesagt vs. Tatsächlich, nach Token-Typ eingefärbt."""
        pred = data["pred_flat"]
        gt   = data["gt_flat"]
        toks = data["token_flat"]

        unique_toks = np.unique(toks)
        tab10 = plt.cm.tab10.colors  # 10 feste Farben

        fig, ax = plt.subplots(figsize=(7, 6))
        for idx, tok in enumerate(unique_toks):
            mask  = toks == tok
            label = INV_VOCAB.get(int(tok), str(tok))
            color = tab10[idx % len(tab10)]
            ax.scatter(gt[mask], pred[mask], alpha=0.4, s=15,
                       color=color, label=label, edgecolors="none")

        min_val = min(gt.min(), pred.min())
        max_val = max(gt.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal")
        ax.set_xlabel("Tatsächlich [min]")
        ax.set_ylabel("Vorhergesagt [min]")
        ax.set_title(f"{title_prefix} – Schrittzeiten: Vorhergesagt vs. Tatsächlich")
        ax.legend(fontsize=8, markerscale=2)
        fig.tight_layout()
        path = os.path.join(tmp, f"{filename_prefix}scatter.png")
        self._log(fig, path, run_id, f"{artifact_dir}/scatter")

    def _plot_per_token_mae(self, data, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        """Mittlerer absoluter Fehler (MAE) pro Prozessschritttyp."""
        pred = data["pred_flat"]
        gt   = data["gt_flat"]
        toks = data["token_flat"]

        # Sonder-Token ausblenden
        special_ids = {VOCAB[k] for k in _SPECIAL_TOKENS if k in VOCAB}
        unique_toks = [int(t) for t in np.unique(toks) if int(t) not in special_ids]

        labels = [INV_VOCAB.get(t, str(t)) for t in unique_toks]
        maes: list[float] = []
        counts: list[int] = []
        for tok in unique_toks:
            mask = toks == tok
            maes.append(float(np.abs(pred[mask] - gt[mask]).mean()) if mask.sum() > 0 else 0.0)
            counts.append(int(mask.sum()))

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.bar(labels, maes, edgecolor="black", color="#4c72b0")
        ax.set_ylabel("MAE [min]")
        ax.set_title(f"{title_prefix} – MAE pro Prozessschritttyp")
        max_mae = max(maes) if maes else 1.0
        for bar, val, cnt in zip(bars, maes, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02 * max_mae,
                f"{val:.1f}\n(n={cnt})",
                ha="center", va="bottom", fontsize=8,
            )
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, max_mae * 1.25)
        fig.tight_layout()
        path = os.path.join(tmp, f"{filename_prefix}token_mae.png")
        self._log(fig, path, run_id, f"{artifact_dir}/token_mae")

    def _plot_consistency(self, data, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        """Summe der vorhergesagten Schrittzeiten vs. GT-Gesamtzeit (Konsistenz-Check)."""
        pred_total = data["pred_total"]
        gt_total   = data["gt_total"]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(gt_total, pred_total, alpha=0.4, edgecolors="none", s=20, color="#c44e52")
        min_val = min(gt_total.min(), pred_total.min())
        max_val = max(gt_total.max(), pred_total.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal")
        mae_total = float(np.abs(pred_total - gt_total).mean())
        ax.set_xlabel("GT Gesamtzeit [min]")
        ax.set_ylabel("Σ Vorhergesagte Schrittzeiten [min]")
        ax.set_title(f"{title_prefix} – Konsistenz: Gesamtzeit  (MAE={mae_total:.1f} min)")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(tmp, f"{filename_prefix}consistency.png")
        self._log(fig, path, run_id, f"{artifact_dir}/consistency")

    def _plot_error_distribution(self, data, title_prefix, tmp, run_id, artifact_dir, filename_prefix):
        """Verteilung der per-Schritt-Fehler als Histogramm mit KDE."""
        errors = data["pred_flat"] - data["gt_flat"]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(errors, ax=ax, kde=True, color="#4c72b0", alpha=0.6)
        ax.axvline(errors.mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mittelwert: {errors.mean():.2f} min")
        ax.axvline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)
        ax.set_xlabel("Fehler [min] (Vorhergesagt – Tatsächlich)")
        ax.set_ylabel("Häufigkeit")
        ax.set_title(f"{title_prefix} – Fehlerverteilung (Schrittzeiten)")
        ax.legend()
        fig.tight_layout()
        path = os.path.join(tmp, f"{filename_prefix}errors.png")
        self._log(fig, path, run_id, f"{artifact_dir}/errors")


# ---------------------------------------------------------------------------
# Epochenweise StepTime-Plots
# ---------------------------------------------------------------------------

class StepTimePredictionPlotCallback(_StepTimePlotMixin, Callback):
    """Loggt vier Schrittzeit-Diagnose-Plots alle ``plot_every_n_epochs`` Epochen.

    Plots:
    * Scatter (pred vs. gt, nach Token eingefärbt)
    * MAE pro Prozessschritttyp
    * Konsistenz-Scatter (Σ pred vs. GT-Gesamtzeit)
    * Fehlerverteilung

    Parameters
    ----------
    plot_every_n_epochs : int
        Frequenz der Plot-Erzeugung.
    """

    def __init__(self, plot_every_n_epochs: int = 10):
        self.plot_every_n_epochs = plot_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.plot_every_n_epochs != 0:
            return
        if not isinstance(trainer.logger, MLFlowLogger):
            return

        val_dl = trainer.val_dataloaders
        if isinstance(val_dl, list):
            val_dl = val_dl[0]

        data = self._collect_predictions(val_dl, pl_module)

        epoch = trainer.current_epoch
        self._generate_plots(
            data,
            title_prefix=f"Epoch {epoch}",
            run_id=trainer.logger.run_id,
            artifact_dir="plots",
            filename_prefix=f"ep{epoch:04d}_",
        )


# ---------------------------------------------------------------------------
# Best-Model StepTime-Plots (werden überschrieben)
# ---------------------------------------------------------------------------

class BestStepTimeModelPlotCallback(_StepTimePlotMixin, Callback):
    """Loggt vier Schrittzeit-Diagnose-Plots für das aktuell beste Modell.

    Die Dateien liegen unter ``plots/best/`` und werden bei jedem neuen
    Checkpoint-Bestwert überschrieben.
    """

    def __init__(self):
        self._last_best_path: str = ""

    def on_validation_epoch_end(self, trainer, pl_module):
        if not isinstance(trainer.logger, MLFlowLogger):
            return

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

        data = self._collect_predictions(val_dl, pl_module)

        epoch = trainer.current_epoch
        logger.info(
            f"Neues bestes Modell (Epoch {epoch}) – StepTime-Best-Plots werden überschrieben."
        )
        self._generate_plots(
            data,
            title_prefix=f"Bestes Modell – Epoch {epoch}",
            run_id=trainer.logger.run_id,
            artifact_dir="plots/best",
            filename_prefix="",
        )
