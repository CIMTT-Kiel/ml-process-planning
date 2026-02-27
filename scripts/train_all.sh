#!/usr/bin/env bash
# ============================================================
# Sequenzielles Training aller ML-Pipelines
#
# Ausführung mit nohup:
#   nohup bash scripts/train_all.sh > logs/train_all.log 2>&1 &
#   tail -f logs/train_all.log
#
# Das Skript bricht ab, wenn eine Pipeline mit einem Fehler
# endet (set -e). Logs werden mit Zeitstempel versehen.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "=== Starte sequenzielles Training ==="
log "Projektverzeichnis: $PROJECT_ROOT"
log ""

# ----------------------------------------------------------
# Pipeline 1: MTL Zeit + Kosten pro Schritt
# ----------------------------------------------------------
log ">>> Pipeline 1/3: time-and-cost-per-step"
python -m "mpp.ml.pipelines.time-and-cost-per-step.model_input_to_tuned_model"
log "<<< Pipeline 1/3 abgeschlossen"
log ""

# ----------------------------------------------------------
# Pipeline 2: CAD-zu-Sequenz
# ----------------------------------------------------------
log ">>> Pipeline 2/3: cadtoseq"
python -m "mpp.ml.pipelines.cadtoseq.model_input_to_tuned_model"
log "<<< Pipeline 2/3 abgeschlossen"
log ""

# ----------------------------------------------------------
# Pipeline 3: Schrittzeit-Regression
# ----------------------------------------------------------
log ">>> Pipeline 3/3: step-time-regression"
python -m "mpp.ml.pipelines.step-time-regression.model_input_to_tuned_model"
log "<<< Pipeline 3/3 abgeschlossen"
log ""

log "=== Alle Pipelines erfolgreich abgeschlossen ==="
