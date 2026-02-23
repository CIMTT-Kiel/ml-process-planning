"""
Erzeugt Vecsets (features/vecset.npy) aus STEP-Dateien.
Ausführung: sudo -E python -m mpp.ml.datasets.preprocessing.create_invariants
"""

import logging
import os
import sys

from tqdm import tqdm
from client.core import CADConverterClient
from mpp.constants import PATHS


class _TqdmHandler(logging.StreamHandler):
    """Leitet Log-Ausgaben durch tqdm.write(), damit der Balken nicht zerstört wird."""
    def emit(self, record):
        tqdm.write(self.format(record))


# Nur den eigenen Logger aktiv lassen, alle anderen Bibliotheken stumm schalten
logging.root.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = _TqdmHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(_handler)
logger.propagate = False

if __name__ == "__main__":
    if os.getuid() != 0:
        logger.warning("Nicht als Root gestartet – Schreibrechte könnten fehlen. Starte mit: sudo -E ...")

    samples = sorted(p for p in PATHS.PP_DATA.iterdir() if p.is_dir() and p.name != ".DS_Store")
    logger.info(f"{len(samples)} Samples gefunden in {PATHS.PP_DATA}")

    converter = CADConverterClient()
    logger.info(converter.get_service_status())

    ok, fail = 0, 0
    for sample_dir in tqdm(samples, desc="Vecsets erzeugen", unit="Sample"):
        idx = sample_dir.name
        step_file    = sample_dir / f"geometry_{idx}.STEP"
        features_dir = sample_dir / "features"
        vecset_path  = features_dir / "vecset.npy"

        if vecset_path.exists():
            ok += 1
            continue

        if not step_file.exists():
            tqdm.write(f"[SKIP] {idx} – kein STEP-File")
            ok += 1
            continue

        try:
            features_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(features_dir, 0o755)

            converter.convert_to_vecset(step_file, vecset_path)
            os.chmod(vecset_path, 0o644)

            ok += 1
        except Exception as e:
            tqdm.write(f"[FAIL] {idx} – {e}")
            fail += 1

    logger.info(f"Fertig. OK: {ok}  Fehler: {fail}  Gesamt: {len(samples)}")
    if fail > 0:
        sys.exit(1)
