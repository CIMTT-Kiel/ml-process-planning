"""
This module defines several constants used throughout the project.

It provides centralized access to project-wide paths (e.g., data, reports, models)
as well as vocabulary definitions for process steps.

Constants are grouped using NamedTuple objects for convenient attribute-based access.

Pfade für Rohdaten können über Umgebungsvariablen überschrieben werden:
    MPP_PP_DATA      – Pfad zu den Produktionsplan-Daten (fabricad)
    MPP_FEATURE_DATA – Pfad zu den Feature-Daten (vecset.npy)

Examples
--------
>>> from mpp import constants
>>> constants.PATHS.ROOT  # Path to the project root
>>> constants.VOCAB["bohren"]  # Get token ID for a process step
"""
# standard imports
import os
from pathlib import Path
from collections import namedtuple

# -------------------------------
# Define project-wide filesystem paths
# -------------------------------

# Determine ROOT (two levels up from this file: src/mpp/ -> src/ -> ROOT)
_ROOT = Path(__file__).parents[2]

# Dictionary of relevant project paths
_path_dict = {
    "ROOT":           _ROOT,
    "REPORT":         _ROOT / "reports",
    "REPORT_FIGURES": _ROOT / "reports/figures",
    "CONFIG":         _ROOT / "config",

    "CKPT_DIR":       _ROOT / "src/mpp/ml/models/checkpoints",
    "MODEL_DIR":      _ROOT / "models",

    # Datenpfade: über Umgebungsvariablen konfigurierbar
    # Beispiel: export MPP_PP_DATA=/data/fabricad
    "PP_DATA":      Path(os.environ.get("MPP_PP_DATA", "/home/coder/shared/datasets/fabricad/fabricad-100k")),
    "FEATURE_DATA": Path(os.environ.get("MPP_FEATURE_DATA", "/data/feature")),
}

# -------------------------------
# Define token vocabulary
# -------------------------------

# Vocabulary for manufacturing process steps and special tokens
VOCAB = {
    "fräsen": 0,
    "schleifen": 1,
    "bohren": 2,
    "schweißen": 3,
    "drehen": 4,
    "prüfen": 5,
    "kontrollieren": 6,
    "START": 7,
    "STOP": 8,
    "PAD": 9,
}

# check if all keys and values are unique
assert len(VOCAB) == len(set(VOCAB.values())), ValueError("VOCAB values must be unique")
assert len(VOCAB) == len(set(VOCAB.keys())), ValueError("VOCAB keys must be unique")

# Inverted vocabulary: maps token IDs back to string labels
INV_VOCAB = {v: k for k, v in VOCAB.items()}

# Convert path dictionary to a namedtuple for attribute-style access
Paths = namedtuple("Paths", list(_path_dict.keys()))
PATHS = Paths(**_path_dict)

# clean up
del _path_dict
del Paths
del _ROOT
del namedtuple
del Path
del os
