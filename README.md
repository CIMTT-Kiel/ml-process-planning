
# ML Process Planning (MPP)

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org)

A machine learning framework for manufacturing process planning, developed by the Center for Industrial Manufacturing Technology and Transfer (CIMTT) at Kiel University of Applied Sciences. This project focuses on predicting manufacturing sequences, process steps, and time estimation from CAD data using deep learning approaches.

## 🎯 Project Overview

This repository implements machine learning models for manufacturing process planning, specifically:

- **CAD to Process Sequence Prediction**: Transform CAD models into manufacturing process sequences
- **CAD to Multi-label process-classification**: Predict sets of processing steps to manufactur a part
- **Process Time Regression**: Estimate manufacturing time requirements
- **Process Cost Regression**: Estimate manufacturing Cost requirements - upcomming

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
│       │   └── process_time_regression.yaml # Config for time regression
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
│       │   │   │   ├── process_time_regressor.py     # Time regression Lightning module
│       │   │   │   └── trsfm_encoder_regressor.py    # Transformer-based regressor
│       │   │   ├── sequence/      # Sequence prediction models
│       │   │   │   ├── cadtoseq_module.py      # CAD-to-sequence pipeline
│       │   │   │   └── vecset_transformer.py   # Vector set transformer
│       │   │   └── checkpoints/   # Trained model checkpoints
│       │   │       ├── best_model/       # tuned models pool
│       │   │       └── tuning/           # Hyperparameter tuning results
│       │   └── pipelines/         # Training and inference pipelines
│       │       ├── base_pipeline.py             # Shared utilities (Trainer, Logger, Callbacks)
│       │       ├── cadtoseq/             # Sequence prediction pipelines
│       │       ├── cadtostepset/         # Step classification pipelines
│       │       └── process-time-regression/ # Time regression pipelines
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
- **Architecture**: Transformer encoder with regression head
- **Input**: Vecsets
- **Output**: Continuous time estimate in minutes
- **Loss**: HuberLoss (robust to outliers) on z-score normalized targets
- **Metrics**: MAE and RMSE in absolute minutes

### 4. VoxelEncoder - !Only for Benchmark and testing!
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

# Train time regression model
python -m "mpp.ml.pipelines.process-time-regression.model_input_to_tuned_model"
```

### Hyperparameter Optimization

The project uses Optuna for automated hyperparameter search. Each tuning trial is logged as a nested MLflow run. Results (best checkpoint) are stored under `checkpoints/tuning/`.

## 📈 Model Evaluation

### Metrics

| Approach | Metrics |
|---|---|
| cadtoseq | Levenshtein distance, token-wise accuracy, confusion matrix |
| cadtostepset | Per-class Precision / Recall / F1, Macro-F1 |
| process-time-regression | MAE [min], RMSE [min] |

All metrics and diagnostic plots are logged automatically to MLflow during training.

## 🛠️ Configuration

### YAML-based Configuration System

All training parameters are controlled via YAML files in `src/mpp/config/`. The `base.yaml` defines shared defaults; each task config inherits from it and can override any value.

Key parameters in `base.yaml`:

```yaml
data:
  batch_size: 2048
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

**Author**: Michel Kruse (michel.kruse@fh-kiel.de)

**Organization**: Center for Industrial Manufacturing Technology and Transfer (CIMTT)
Kiel University of Applied Sciences

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support and Contact

- **Issues**: Report bugs via [GitHub Issues](https://github.com/CIMTT-Kiel/ml-process-planning/issues)
- **Email**: michel.kruse@fh-kiel.de
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
