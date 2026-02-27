# MTL-Modul: Zeit und Kosten pro Fertigungsschritt

Dieses Dokument beschreibt das neue `time-and-cost-per-step`-Modul, das
gleichzeitig Dauer und Kosten pro Fertigungsschritt vorhersagt
(Multi-Task Learning mit Kendall-Loss).

---

## Überblick der Änderungen

| Bereich | Art | Detail |
|---|---|---|
| **Ausgabe** | MTL | `[B, seq_len]` Zeit + `[B, seq_len]` Kosten |
| **Target-Typ** | neu | `"step-time-cost"` |
| **Architektur** | neu | Encoder → Decoder mit Zeit-Head + Kosten-Head |
| **Loss** | neu | Kendall-MTL (lernbare Gewichte) + Consistency (normiert) |
| **Phasen** | neu | Phase 1: Encoder eingefroren; Phase 2: diff. LR + Noise |
| **Config** | neu | `config/time_and_cost_regression.yaml` |
| **Pipeline** | neu | `pipelines/time-and-cost-per-step/` |

---

## Neue und geänderte Dateien

### Neu erstellt

```
src/mpp/ml/models/regressor/
├── mtl_step_decoder.py           – MTLStepTimeDecoder, MTLEncoderDecoderModel
└── mtl_step_regression_module.py – MTLStepTimeModule (Lightning-Wrapper)

src/mpp/ml/pipelines/time-and-cost-per-step/
├── __init__.py
└── model_input_to_tuned_model.py – Entry-Point, Optuna-Tuning, Phase1EvaluationCallback

src/mpp/config/time_and_cost_regression.yaml
MIGRATION_MTL.md  – Dieses Dokument
```

### Geändert (additiv, keine Breaking Changes)

```
src/mpp/ml/datasets/fabricad.py
    + _parse_step_time_cost()     – parst Dauer[min] + Kosten[($)] pro Schritt
    + target_type="step-time-cost" in assert und __getitem__

src/mpp/ml/datasets/fabricad_datamodule.py
    + collate_fn_mtl()            – Padding für 6-Tupel-Batches
    + _get_collate_fn()           – wählt collate_fn_mtl für "step-time-cost"

src/mpp/ml/callbacks/artifact_callbacks.py
    + _MTLPlotMixin               – gemeinsame Plot-Logik (6 Plots)
    + MTLPredictionPlotCallback   – Plots alle N Epochen
    + BestMTLModelPlotCallback    – Plots bei neuem Best-Checkpoint
```

---

## Schritt-für-Schritt-Anleitung

### 1. Vortraining des Encoders (empfohlen)

Falls kein `StepTimeRegressionModule`-Checkpoint vorhanden:

```bash
python -m "mpp.ml.pipelines.step-time-regression.model_input_to_tuned_model"
```

Checkpoint-Pfad notieren:
```
src/mpp/ml/models/checkpoints/best_model/step-time-regression/step-time-regressor-42-0.1234.ckpt
```

### 2. Config anpassen

In `src/mpp/config/time_and_cost_regression.yaml`:

```yaml
training:
  pretrained_encoder_ckpt: "src/mpp/ml/models/checkpoints/best_model/step-time-regression/..."
  embed_dim: 128              # muss mit Checkpoint übereinstimmen
  num_encoder_layers: 4       # muss mit Checkpoint übereinstimmen
```

### 3. MTL-Modell trainieren

```bash
python -m "mpp.ml.pipelines.time-and-cost-per-step.model_input_to_tuned_model"
```

Ablauf:
1. Normalisierungsstatistiken aus Trainingsdaten berechnen (non-PAD Positionen)
2. Optuna-Hyperparameter-Tuning (`tuning_epochs` Epochen pro Trial)
3. Finales Training mit besten Hyperparametern:
   - Phase 1 (`freeze_encoder_epochs` Epochen): Nur Decoder + Kendall-Gewichte
   - Ende Phase 1: Hinweis zur Noise-Kalibrierung in MLflow geloggt
   - Phase 2 (`finetune_epochs` Epochen): Encoder aufgetaut, Noise-Injektion aktiv

### 4. Noise-Kalibrierung (optional, Phase 3)

Nach Phase 1 val-Metriken aus MLflow ablesen:

```
val_mae_time_<token>  → noise_overrides_time.<token> = mae / mean_time
val_mae_cost_<token>  → noise_overrides_cost.<token> = mae / mean_cost
```

Dann in YAML setzen und Modell neu trainieren:

```yaml
training:
  noise_overrides_time:
    schweißen: 0.20    # hohe Varianz
    bohren: 0.05       # geringe Varianz
```

### 5. Inference

```python
from mpp.ml.models.regressor.mtl_step_regression_module import MTLStepTimeModule

module = MTLStepTimeModule.load_from_checkpoint("path/to/best.ckpt")
module.eval()

# Autoregressive Inferenz (gesamter Batch):
pred_t_norm, pred_c_norm = module.model.generate(vecset, step_tokens)
pred_times = module._denormalize_time(pred_t_norm)   # [B, seq_len], Minuten
pred_costs = module._denormalize_cost(pred_c_norm)   # [B, seq_len], Dollar

# Streaming (Streamlit, B=1):
for step_idx, token_id, time_min, cost_dollar in module.model.generate_stream(
    vecset,       # [1, 1024, 32]
    step_tokens,  # [1, seq_len]
    target_mean_time=module.hparams.target_mean_time,
    target_std_time=module.hparams.target_std_time,
    target_mean_cost=module.hparams.target_mean_cost,
    target_std_cost=module.hparams.target_std_cost,
):
    print(f"  {INV_VOCAB[token_id]:>15s}  {time_min:6.1f} min  {cost_dollar:8.2f} $")
```

---

## Neues Datenformat (`target_type="step-time-cost"`)

```
Batch: (
    vecset     [B, 1024, 32],
    step_tokens [B, S],
    step_times  [B, S],      # Dauer[min], PAD-Stellen = 0.0
    step_costs  [B, S],      # Kosten[($)], PAD-Stellen = 0.0
    total_time  [B],         # Summe der gefilterten Schritte
    total_cost  [B],         # Summe der gefilterten Schritte
)
```

**Filterlogik:** Erste Zeile (Index 0) und `"liefern"` werden ausgeblendet –
identisch zu `target_type="step-time"`.

---

## Loss-Funktion

### Kendall-MTL (lernbare Gewichte)

```
L_MTL = exp(-s_t) · Huber_time + s_t
      + exp(-s_c) · Huber_cost + s_c
```

`s_t = log_var_time`, `s_c = log_var_cost` sind trainierbare `nn.Parameter`.

### Consistency-Loss (normierter Raum)

```
L_cons = λ_t · MSE(Σ pred_times_norm, total_time_norm_sum)
       + λ_c · MSE(Σ pred_costs_norm, total_cost_norm_sum)

total_time_norm_sum = (total_time - n_valid · mean_t) / std_t
```

Beide Terme im normierten Raum → Skala vergleichbar mit Huber-Loss.

---

## MLflow-Plots (6 Plots pro Checkpoint)

| Plot | Inhalt |
|---|---|
| `scatter_time` | Pred vs. Tatsächlich [min], nach Token eingefärbt |
| `scatter_cost` | Pred vs. Tatsächlich [$], nach Token eingefärbt |
| `token_mae` | MAE pro Schritttyp (Zeit + Kosten nebeneinander) |
| `consistency_time` | Σ Pred-Zeiten vs. GT-Gesamtzeit |
| `consistency_cost` | Σ Pred-Kosten vs. GT-Gesamtkosten |
| `errors` | Fehlerverteilung (Histogramm + KDE, beide Tasks) |

---

## Kompatibilität

- `step-time-regression`-Pipeline bleibt vollständig unverändert funktionsfähig.
- Bestehende `target_type`-Werte (`"time"`, `"cost"`, `"step-set"`, `"seq"`, `"step-time"`)
  funktionieren wie bisher.
