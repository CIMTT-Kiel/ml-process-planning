
# ML Process Planning (MPP)

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org)

A machine learning framework for manufacturing process planning, developed by the Center for Industrial Manufacturing Technology and Transfer (CIMTT) at Kiel University of Applied Sciences. This project focuses on predicting manufacturing sequences, process steps, and time estimation from CAD data using deep learning approaches.

## 🎯 Project Overview

This repository implements machine learning models for manufacturing process planning, specifically:

- **CAD to Process Sequence Prediction**: Transform CAD models into ordered manufacturing process sequences
- **CAD to Multi-label Process Classification**: Predict which processing steps are required to manufacture a part
- **Process Time Regression**: Estimate total manufacturing time for a part
- **Step-Time Regression**: Predict the duration of each individual manufacturing step (autoregressive encoder-decoder)

The project leverages the FabriCAD dataset and implements transformer-based architectures for sequence-to-sequence learning in manufacturing contexts.

## 🏗️ Project Structure

```
ml-process-planning/
├── LICENSE                          # Project license
├── README.md                        # This file
├── pyproject.toml                   # Project configuration and dependencies

├── notebooks/                      # Jupyter notebooks
├── reports/                        # Documentation and results
│   ├── experiments/               # Experiment documentation
│   └── figures/                   # Generated plots and visualizations
├── src/                           # Source code
│   └── mpp/                       # Main package
│       ├── constants.py           # Project constants and configurations
│       ├── config/                # YAML configuration files
│       │   ├── base.yaml                    # Shared base config (batch_size, gpu_id, ...)
│       │   ├── cadtoseq.yaml                # Config for sequence prediction
│       │   ├── cadtostepset.yaml            # Config for step classification
│       │   ├── process_time_regression.yaml # Config for total-time regression
│       │   └── step_time_regression.yaml    # Config for per-step time regression
│       ├── ml/                    # Machine learning modules
│       │   ├── callbacks/         # PyTorch Lightning callbacks
│       │   │   └── artifact_callbacks.py    # MLflow artifact logging (plots, checkpoints)
│       │   ├── datasets/          # Data loading and preprocessing
│       │   │   ├── fabricad_datamodule.py    # FabriCAD dataset pl-integration
│       │   │   ├── fabricad.py               # FabriCAD pt Dataset
│       │   │   └── tkms_dataset.py           # TKMS dataset support placeholder
│       │   ├── metrics/           # Custom evaluation metrics
│       │   │   └── sequences.py   # Sequence-specific metrics
│       │   ├── models/            # Model implementations
│       │   │   ├── classifier/    # Classification models
│       │   │   │   ├── cadtostepset.py          # Multi-label step classifier
│       │   │   │   ├── multilabel_classifier.py # Generic multi-label classifier
│       │   │   │   └── VoxelEncoder.py          # 3D voxel encoding (outdatated)
│       │   │   ├── regressor/     # Regression models
│       │   │   │   ├── process_time_regressor.py         # Total-time regression Lightning module
│       │   │   │   ├── trsfm_encoder_regressor.py        # Transformer encoder (shared backbone)
│       │   │   │   ├── step_time_decoder.py              # Causal Transformer decoder for step times
│       │   │   │   └── step_time_regression_module.py    # Step-time regression Lightning module
│       │   │   ├── sequence/      # Sequence prediction models
│       │   │   │   ├── cadtoseq_module.py      # CAD-to-sequence pipeline
│       │   │   │   └── vecset_transformer.py   # Vector set transformer
│       │   │   └── checkpoints/   # Trained model checkpoints
│       │   │       ├── best_model/       # tuned models pool
│       │   │       └── tuning/           # Hyperparameter tuning results
│       │   └── pipelines/         # Training and inference pipelines
│       │       ├── base_pipeline.py             # Shared utilities (Trainer, Logger, Callbacks)
│       │       ├── cadtoseq/                    # Sequence prediction pipeline
│       │       ├── cadtostepset/                # Step classification pipeline
│       │       ├── process-time-regression/     # Total-time regression pipeline
│       │       └── step-time-regression/        # Per-step time regression pipeline
└── tests/                         # for unit tests - not implemented yet
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended for training)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/CIMTT-Kiel/ml-process-planning.git
   cd ml-process-planning
   ```

2. **Install using uv (recommended)**
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project dependencies
   uv sync
   ```

3. **Alternative: Install with pip**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install project
   pip install -e .
   ```

### Development Installation

For development with additional tools:
```bash
uv sync --extra dev --extra notebook
```

This installs additional dependencies for:
- **dev**: Testing, linting, and code formatting tools
- **notebook**: Jupyter Lab and kernel support

## 🧠 Machine Learning Models

### 1. CAD-to-Sequence Prediction (`cadtoseq`)
- **Purpose**: Generate complete manufacturing process sequences from CAD input
- **Architecture**: Transformer-based sequence-to-sequence model
- **Input**: Vecsets
- **Output**: Ordered sequence of standard manufacturing operations
- **Metrics**: Levenshtein distance, token-wise accuracy, confusion matrix

### 2. CAD-to-Step Classification (`cadtostepset`)
- **Purpose**: Predict required manufacturing steps (multi-label classification)
- **Architecture**: Transformer encoder with multi-label classification head
- **Input**: Vecsets
- **Output**: Set of required manufacturing steps (no ordering, no repetition)
- **Loss**: BCEWithLogitsLoss with per-class `pos_weight` to handle class imbalance
- **Metrics**: Per-class Precision, Recall, F1; Macro-F1

### 3. Process Time Regression (`process-time-regression`)
- **Purpose**: Estimate total manufacturing time for a part
- **Architecture**: Transformer encoder + mean-pooling + regression head
- **Input**: Vecsets
- **Output**: Single continuous time estimate in minutes
- **Loss**: HuberLoss on z-score normalized targets
- **Metrics**: MAE and RMSE in absolute minutes

### 4. Step-Time Regression (`step-time-regression`)
- **Purpose**: Predict the duration of each individual manufacturing step
- **Architecture**: Transformer encoder (shared backbone from model 3) + causal Transformer decoder
- **Input**: Vecsets + step token sequence (e.g. from `cadtoseq`)
- **Output**: Per-step time in minutes `[B, seq_len]`, autoregressive generation
- **Loss**: HuberLoss per step (normalized) + λ · MSE(Σ predicted steps, total time) – both in normalized space
- **Training**: Two phases – Phase 1 encoder frozen (decoder only), Phase 2 differential learning rates
- **Inference**: Teacher forcing (validation) or autoregressive `generate()` / streaming `generate_stream()`
- **Metrics**: MAE and RMSE per step, consistency MAE (total time)

### 5. VoxelEncoder — benchmark only
- **Purpose**: Encode 3D CAD data into feature representations
- **Architecture**: 3D convolutional neural network
- **Input**: Voxelized CAD models
- **Output**: Dense feature vectors

## 📊 Datasets

The project supports multiple manufacturing datasets:

### FabriCAD Integration
- **Source**: CIMTT's synthetic manufacturing dataset
- **Content**: CAD models paired with process plans
- **Format**: STEP files + CSV process descriptions
- **Access**: Via `fabricad_datamodule.py` and `fabricad.py`

### TKMS Dataset
- **Purpose**: Additional real-world manufacturing process data
- **Integration**: Through `tkms_dataset.py`

## 🔬 Experimentation and Training

### MLflow Experiment Tracking

The project uses a remote MLflow server for experiment tracking (configured in `config/base.yaml`):

```yaml
mlflow:
  tracking_uri: "http://mlflow-server:5000"
```

Each training run logs hyperparameters, metrics, plots, and model checkpoints automatically.

### Training Pipelines

Each model type has its dedicated training pipeline that runs Optuna hyperparameter tuning followed by a final training run with the best configuration:

```bash
# Train CAD-to-sequence model
python -m mpp.ml.pipelines.cadtoseq.model_input_to_tuned_model

# Train step classification model
python -m mpp.ml.pipelines.cadtostepset.model_input_to_tuned_model

# Train total-time regression model
python -m "mpp.ml.pipelines.process-time-regression.model_input_to_tuned_model"

# Train per-step time regression model
python -m "mpp.ml.pipelines.step-time-regression.model_input_to_tuned_model"
```

The step-time pipeline optionally warm-starts the encoder from a pretrained process-time checkpoint. Set `training.pretrained_encoder_ckpt` in `step_time_regression.yaml` accordingly (see `MIGRATION.md` for details).

### Hyperparameter Optimization

The project uses Optuna for automated hyperparameter search. Each tuning trial is logged as a nested MLflow run. Results (best checkpoint) are stored under `checkpoints/tuning/`.

## 🔍 Inference

All models are loaded via PyTorch Lightning's `load_from_checkpoint`. Hyperparameters (including normalization statistics) are restored automatically.

**Total-time regression:**
```python
from mpp.ml.models.regressor.process_time_regressor import ProcessRegressionModule

model = ProcessRegressionModule.load_from_checkpoint("path/to/checkpoint.ckpt")
model.eval()
pred_norm = model(vecset)                          # [B], normalized
pred_min  = pred_norm * model.hparams.target_std + model.hparams.target_mean
```

**Per-step time regression — full batch:**
```python
from mpp.ml.models.regressor.step_time_regression_module import StepTimeRegressionModule

module = StepTimeRegressionModule.load_from_checkpoint("path/to/checkpoint.ckpt")
module.eval()
pred_norm = module.model.generate(vecset, step_tokens)   # [B, seq_len], normalized
pred_min  = module._denormalize(pred_norm)               # [B, seq_len], minutes
```

**Per-step time regression — streaming (e.g. Streamlit):**
```python
from mpp.constants import INV_VOCAB

for step_idx, token_id, time_min in module.model.generate_stream(
    vecset,       # [1, 1024, 32]
    step_tokens,  # [1, seq_len]
    target_mean=module.hparams.target_mean,
    target_std=module.hparams.target_std,
):
    print(f"  {INV_VOCAB[token_id]:>15s}  {time_min:6.1f} min")
```

`generate_stream` yields `(step_index, token_id, time_minutes)` one step at a time, enabling live UI updates while inference is still running.

## 📈 Model Evaluation

### Metrics

| Approach | Metrics | MLflow plots |
|---|---|---|
| `cadtoseq` | Levenshtein distance, token-wise accuracy, exact match | Prediction table, confusion matrix, Levenshtein distribution, token accuracy |
| `cadtostepset` | Per-class Precision / Recall / F1, Macro-F1 | Prediction table, class metric bars |
| `process-time-regression` | MAE [min], RMSE [min] | Scatter (pred vs. actual), residuals, error distribution |
| `step-time-regression` | MAE per step [min], RMSE per step [min], consistency MAE [min] | Scatter per token type, MAE per step type, consistency scatter (Σ steps vs. total), error distribution |

All metrics and diagnostic plots are logged automatically to MLflow during training.

## 🛠️ Configuration

### YAML-based Configuration System

All training parameters are controlled via YAML files in `src/mpp/config/`. The `base.yaml` defines shared defaults; each task config inherits from it and can override any value.

Key parameters in `base.yaml`:

```yaml
data:
  batch_size: 800
  num_workers: 0

training:
  n_trials: 35          # Optuna tuning trials
  tuning_epochs: 50
  final_epochs: 1000
  tuning_patience: 20
  final_patience: 30
  gpu_id: 0             # GPU index (0 or 1)
  weight_decay: 0.01
```

Additional parameters specific to `step_time_regression.yaml`:

```yaml
training:
  freeze_encoder_epochs: 20       # Phase 1: encoder frozen, decoder only
  encoder_lr_factor: 0.1          # Phase 2: encoder LR = lr × factor
  lambda_consistency: 0.1         # weight for Σ-step vs. total-time loss
  scheduled_sampling: false       # use own predictions as decoder input
  scheduled_sampling_rate: 0.5    # fraction of batches using scheduled sampling
  # pretrained_encoder_ckpt: "src/mpp/ml/models/checkpoints/best_model/time-regression/..."
```

To run a task on a specific GPU, set `gpu_id` in the corresponding task config:

```yaml
# e.g. cadtostepset.yaml
training:
  gpu_id: 1
```

### Constants and Settings
Global configurations are managed in `src/mpp/constants.py`:
- paths and parameters
- Token dictionaries


### Code Structure Guidelines

- **Models**: Place new architectures in `src/mpp/ml/models/`
- **Datasets**: Add data loaders to `src/mpp/ml/datasets/`
- **Pipelines**: Create training scripts in `src/mpp/ml/pipelines/`
- **Metrics**: Implement evaluation metrics in `src/mpp/ml/metrics/`

## 📚 Dependencies

### Core Dependencies
- **torch**: Deep learning framework
- **pytorch-lightning**: Training orchestration
- **hydra-core**: Configuration management
- **mlflow**: Experiment tracking
- **scikit-learn**: Machine learning utilities
- **optuna**: Hyperparameter optimization

### Data Processing
- **pandas**: Data manipulation
- **torchvision**: Computer vision utilities

### Visualization
- **matplotlib**: Plotting library
- **seaborn**: Statistical visualization

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **ruff**: Fast linting
- **mypy**: Type checking

## 👥 Team

**Author**: Michel Kruse (michel.kruse@haw-kiel.de)

**Organization**: Center for Industrial Manufacturing Technology and Transfer (CIMTT)
Kiel University of Applied Sciences

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support and Contact

- **Email**: michel.kruse@haw-kiel.de
- **Organization**: CIMTT, FH Kiel

## 🔗 Related Projects

- **[FabriCAD](https://github.com/CIMTT-Kiel/FabriCAD)**: Synthetic manufacturing dataset used in this project

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{ml_process_planning,
  title = {ML Process Planning: Deep Learning for Manufacturing Process Prediction},
  author = {Michel Kruse},
  organization = {CIMTT, Kiel University of Applied Sciences},
  year = {2024},
  url = {https://github.com/CIMTT-Kiel/ml-process-planning}
}
```

---

**Developed by CIMTT at Kiel University of Applied Sciences**
