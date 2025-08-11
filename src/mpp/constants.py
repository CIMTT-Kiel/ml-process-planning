"""
This module defines several constants used throughout the project.

The module provides access to NamedTupel objects which group all constants and make them available via attribute access.

Examples
--------
>>> from project import constants
>>> # get Path object for the project root directory
>>> constants.PATHS.ROOT
"""
from pathlib import Path
from collections import namedtuple


# Paths
_ROOT = Path(__file__).parents[2]
_path_dict = {
    "ROOT":                 _ROOT,
    "REPORT":       _ROOT / "reports",
    "REPORT_FIGURES":       _ROOT / "reports/figures",
    "CONFIG":               _ROOT / "config",

    "CKPT_DIR":            _ROOT / "src/cadtoseq/ml/models/checkpoints",
    "MODEL_DIR":           _ROOT / "models",

    "PP_DATA":      Path("/home/michelkruse/data_repos/fabricad"), #Productplan (PP) data
    "FEATURE_DATA":           Path("/home/michelkruse/repos/FabriCAD/data/4_feature"),

}

VOCAB = {
    "START": 0,
    "fräsen": 1,
    "schleifen": 2,
    "bohren": 3,
    "schweißen": 4,
    "drehen": 5,
    "prüfen":6,
    "kontrollieren": 7,
    "STOP": 8,
    "PAD": 9,
}

INV_VOCAB = {v: k for k, v in VOCAB.items()}



Paths = namedtuple("Paths", list(_path_dict.keys()))
PATHS = Paths(**_path_dict)

# clean up for paths constants
del _path_dict
del Paths
del _ROOT

# general clean up
del namedtuple
del Path