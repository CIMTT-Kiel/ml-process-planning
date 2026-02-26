# Schrittweise Zeitvorhersage

Dieses Dokument beschreibt, wie das bestehende `ProcessRegressionModule`
(ein Skalar pro Probe) auf das neue `StepTimeRegressionModule` (eine Zeit
pro Fertigungsschritt, autoregressive Dekodierung) migriert wird.

---

## Überblick der Änderungen

| Bereich | Alt | Neu |
|---|---|---|
| **Ausgabe** | `[B]` – eine Gesamtzeit | `[B, seq_len]` – eine Zeit pro Schritt |
| **Target-Typ** | `"time"` | `"step-time"` |
| **Architektur** | Encoder → Mean-Pooling → Linear | Encoder (Memory) + Decoder (Causal Cross-Attention) |
| **Loss** | HuberLoss | HuberLoss (pro Schritt) + λ · MSE(Σ pred, total_gt) |
| **Training-Modus** | — | Teacher Forcing / Scheduled Sampling |
| **Training-Phasen** | Eine Phase | Phase 1 (Encoder eingefroren) + Phase 2 (diff. LR) |
| **Config-Datei** | `process_time_regression.yaml` | `step_time_regression.yaml` |
| **Pipeline** | `pipelines/process-time-regression/` | `pipelines/step-time-regression/` |

---

## Neue und geänderte Dateien

### Neu erstellt

```
src/mpp/ml/models/regressor/step_time_decoder.py
    StepTimeDecoder              – Transformer-Decoder mit Causal Masking
    TrsfmEncoderStepTimeModel    – Kombination Encoder + Decoder

src/mpp/ml/models/regressor/step_time_regression_module.py
    StepTimeRegressionModule     – Lightning-Wrapper (2-Phasen, Huber+Consistency)

src/mpp/ml/pipelines/step-time-regression/model_input_to_tuned_model.py
    – Optuna-Tuning + finales Training (analog zur process-time-regression-Pipeline)

src/mpp/config/step_time_regression.yaml
    – Alle neuen Hyperparameter

MIGRATION.md
    – Dieses Dokument
```

### Geändert (additiv, keine Breaking Changes)

```
src/mpp/ml/models/regressor/trsfm_encoder_regressor.py
    + encode()  → gibt [B, set_size, embed_dim] vor Mean-Pooling zurück
      forward() nutzt intern encode() (kein Verhaltensunterschied)

src/mpp/ml/datasets/fabricad.py
    + target_type="step-time"  → gibt (step_tokens, step_times, total_time) zurück
    + _parse_step_time()        – interne Hilfsmethode

src/mpp/ml/datasets/fabricad_datamodule.py
    + collate_fn_step_time()    – Padding für 4-Tupel-Batches
    + _get_collate_fn()         – wählt collate-Funktion je target_type
      train/val/test_dataloader() vereinfacht (keine if/else-Wiederholung)
```

---

## Schritt-für-Schritt-Anleitung

### 1. Bestehenden Encoder vortrainieren (optional, empfohlen)

Falls noch kein trainiertes `ProcessRegressionModule`-Checkpoint vorhanden ist,
zuerst das bestehende Regressions-Modell trainieren:

```bash
python -m "mpp.ml.pipelines.process-time-regression.model_input_to_tuned_model"
```

Den Pfad zum besten Checkpoint notieren, z. B.:
```
src/mpp/ml/models/checkpoints/best_model/time-regression/time-regressor-42-0.1234.ckpt
```

### 2. Config anpassen

In `src/mpp/config/step_time_regression.yaml` den Checkpoint-Pfad setzen
(Zeile auskommentieren):

```yaml
training:
  pretrained_encoder_ckpt: "src/mpp/ml/models/checkpoints/best_model/time-regression/time-regressor-42-0.1234.ckpt"
```

> **Wichtig:** `embed_dim` und `num_encoder_layers` im `hyperparameter_search`-
> Abschnitt müssen den Werten des vortrainierten Modells entsprechen.  Wenn das
> vortrainierte Modell z. B. `embed_dim=192, num_encoder_layers=3` hat, müssen
> die Search-Bounds `low=high=192` bzw. `low=high=3` gesetzt werden, damit
> Optuna immer die passende Architektur wählt.

### 3. Schrittweise-Zeitvorhersage trainieren

```bash
python -m "mpp.ml.pipelines.step-time-regression.model_input_to_tuned_model"
```

Der Ablauf:

1. **Statistiken**: Mean/Std der per-Schritt-Zeiten (nur valide, nicht-PAD) werden
   aus dem Trainingsset berechnet.
2. **Phase 1** (Epochen 0 … `freeze_encoder_epochs − 1`): Encoder eingefroren,
   nur Decoder wird trainiert.
3. **Phase 2** (ab Epoche `freeze_encoder_epochs`): Encoder mit LR
   `lr × encoder_lr_factor` mittrainiert.

### 4. Inference (autoregressive Dekodierung)

```python
from mpp.ml.models.regressor.step_time_regression_module import StepTimeRegressionModule

module = StepTimeRegressionModule.load_from_checkpoint("path/to/best.ckpt")
module.eval()

# step_tokens: [B, seq_len] – vorhergesagte Schrittsequenz (z. B. aus ARMSTM)
pred_times_norm = module.model.generate(vecset, step_tokens)  # [B, seq_len]
pred_times_min  = module._denormalize(pred_times_norm)         # in Minuten
```

---

## Neue Hyperparameter in der Config

| Parameter | Typ | Standard | Beschreibung |
|---|---|---|---|
| `training.freeze_encoder_epochs` | int | `20` | Epochen, in denen der Encoder eingefroren bleibt (Phase 1) |
| `training.encoder_lr_factor` | float | `0.1` | Encoder-LR = `lr × factor` in Phase 2 |
| `training.lambda_consistency` | float | `0.1` | Gewicht λ für den Consistency-Loss |
| `training.scheduled_sampling` | bool | `false` | Scheduled Sampling aktivieren |
| `training.scheduled_sampling_rate` | float | `0.5` | Anteil der SS-Batches pro Epoche |
| `training.pretrained_encoder_ckpt` | str \| null | `null` | Pfad zu vortrainiertem Checkpoint |

---

## Datenformat

### Altes Format (`target_type="time"`)

```
Batch: (vecset [B,1024,32], total_time [B])
```

### Neues Format (`target_type="step-time"`)

```
Batch: (vecset [B,1024,32], step_tokens [B,S], step_times [B,S], total_time [B])

S      – Sequenzlänge (variabel, gepaddet auf max. Länge im Batch)
PAD    – step_tokens == VOCAB["PAD"] markiert ungültige Positionen
         step_times == 0.0 an PAD-Positionen
```

### Herkunft von `total_time`

`total_time` ist die **Summe der gefilterten Schritte** (erster Schritt und
`"liefern"` werden wie in `target_type="seq"` ausgeblendet).  Damit stimmt
sie mit der Summe der vorherzusagenden `step_times` überein und der
Consistency-Loss ist exakt.

> Dies weicht vom `"time"`-Target ab, das *alle* Zeilen der `plan.csv`
> summiert.  Der Unterschied beträgt in der Praxis meist wenige Minuten
> (Anfangs- und Lieferschritte haben häufig kurze Dauern).

---

## Design-Entscheidungen (abweichend vom Plan)

| # | Entscheidung | Begründung |
|---|---|---|
| 1 | `encode()` statt neuem separaten Encoder | Minimale Änderung am bestehenden `TrsfmEncoderRegressor`; `forward()` bleibt identisch → keine Breaking Changes |
| 2 | `total_time` = Summe gefilterter Schritte | Ermöglicht exakten Consistency-Loss; im Kommentar dokumentiert |
| 3 | Zwei Parametergruppen im Optimizer (nicht Optimizer-Neustart) | PyTorch Lightning empfiehlt Single-Optimizer; Parameter werden in Phase 1 via `requires_grad=False` eingefroren, sodass der Optimizer sie ignoriert |
| 4 | Scheduled Sampling: konstante Rate (kein Curriculum) | Einfachere Implementierung; für Curriculum-Scheduled-Sampling kann `scheduled_sampling_rate` in einem Custom-Callback pro Epoche angepasst werden |
| 5 | Normalisierung auf per-Schritt-Basis (nicht total) | Konsistenter mit dem HuberLoss; Consistency-Loss arbeitet in absoluten Minuten |
