#!/usr/bin/env python3
"""Ersetzt defekte fabricad-100k Datenpunkte mit validen Daten aus dem Backlog.

Die Ziel-Ordner sind schreibgeschützt, daher muss das Script mit sudo-Rechten
ausgeführt werden. Nach dem Kopieren werden die Berechtigungen auf den Standard
des Datensatzes zurückgesetzt (Ordner: 755, Dateien: 644, Owner: root:root).

Überflüssige Dateien aus dem Backlog (interim/interim_*.png, interim/interim_*.STEP)
werden entfernt, da sie nicht zur fabricad-100k Struktur gehören.
Vorhandene features/-Ordner in den Ziel-Samples bleiben unangetastet.

Verwendung:
    sudo python3 repair_broken_samples.py [--dry-run]
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# =============================================================================
# Konfiguration
# =============================================================================

DATASET_DIR = Path("/home/coder/shared/datasets/fabricad/fabricad-100k")
BACKLOG_DIR = Path("/home/coder/workspace/ml-process-planning/tmp_backlog_data")

# Defekte Sample-IDs (Ziele), sortiert
BROKEN_IDS = [
    "00000599",
    "00001394",
    "00011301",
    "00013002",
    "00015416",
    "00020021",
    "00022076",
    "00024226",
    "00028874",
    "00036725",
    "00041319",
    "00041890",
    "00046463",
    "00048694",
    "00060233",
    "00065858",
    "00069904",
    "00071657",
    "00077229",
    "00078340",
    "00080669",
    "00087413",
    "00091555",
    "00094325",
    "00099095",
    "00100205",
]

# Dateimuster in interim/, die NICHT zur fabricad-100k Struktur gehören
EXTRA_INTERIM_PATTERNS = ["interim_*.png", "interim_*.STEP", "interim_*.step"]

# Berechtigungen passend zum bestehenden Datensatz
DIR_MODE  = 0o755  # rwxr-xr-x
FILE_MODE = 0o644  # rw-r--r--


# =============================================================================
# Hilfsfunktionen
# =============================================================================

def get_backlog_samples() -> list[str]:
    """Gibt sortierte Liste der verfügbaren Backlog-Sample-IDs zurück."""
    return sorted(p.name for p in BACKLOG_DIR.iterdir() if p.is_dir())


def set_permissions(path: Path) -> None:
    """Setzt Berechtigungen rekursiv: Ordner 755, Dateien 644.
    Der features/-Ordner wird übersprungen (wird separat befüllt).
    """
    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        if root_path.name == "features":
            dirs.clear()  # os.walk nicht weiter in features/ eintauchen
            continue
        os.chmod(root_path, DIR_MODE)
        for fname in files:
            os.chmod(root_path / fname, FILE_MODE)


def repair_sample(target_id: str, source_id: str, dry_run: bool = False) -> bool:
    """Ersetzt den Inhalt von target_id mit dem Inhalt von source_id.

    Returns:
        True wenn erfolgreich, False bei Fehler.
    """
    target_dir = DATASET_DIR / target_id
    source_dir = BACKLOG_DIR / source_id

    print(f"  [{target_id}] <- {source_id}")

    if not target_dir.exists():
        print(f"    FEHLER: Ziel-Ordner nicht gefunden: {target_dir}")
        return False
    if not source_dir.exists():
        print(f"    FEHLER: Quell-Ordner nicht gefunden: {source_dir}")
        return False

    if dry_run:
        extra_files = []
        interim_src = source_dir / "interim"
        if interim_src.exists():
            for pattern in EXTRA_INTERIM_PATTERNS:
                extra_files.extend(f.name for f in interim_src.glob(pattern))

        print(f"    [DRY RUN] Lösche Inhalt von {target_dir} (außer features/)")
        print(f"    [DRY RUN] Kopiere von {source_dir}")
        print(f"    [DRY RUN] Umbenennen: geometry_{source_id}.STEP → geometry_{target_id}.STEP")
        if extra_files:
            print(f"    [DRY RUN] Entferne Extra-Dateien: {extra_files}")
        print(f"    [DRY RUN] Leere negative/-Ordner")
        print(f"    [DRY RUN] Setze Berechtigungen (Ordner 755, Dateien 644)")
        return True

    # ------------------------------------------------------------------
    # Schritt 1: Bestehenden Inhalt löschen (features/ bleibt erhalten)
    # ------------------------------------------------------------------
    for item in target_dir.iterdir():
        if item.name == "features":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    # ------------------------------------------------------------------
    # Schritt 2: Inhalt aus Backlog kopieren
    # ------------------------------------------------------------------
    for item in source_dir.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # ------------------------------------------------------------------
    # Schritt 3: Geometry-Datei auf Ziel-ID umbenennen
    # ------------------------------------------------------------------
    geo_src = target_dir / f"geometry_{source_id}.STEP"
    geo_dst = target_dir / f"geometry_{target_id}.STEP"
    if geo_src.exists():
        geo_src.rename(geo_dst)
    else:
        # Fallback: erste geometry_*.STEP umbenennen die noch nicht die Ziel-ID hat
        candidates = [
            f for f in target_dir.glob("geometry_*.STEP")
            if f.name != f"geometry_{target_id}.STEP"
        ]
        if candidates:
            candidates[0].rename(geo_dst)
            print(f"    HINWEIS: '{candidates[0].name}' umbenannt (abweichende Quell-ID im Backlog).")
        elif not geo_dst.exists():
            print(f"    WARNUNG: Keine geometry_*.STEP Datei in {target_dir} gefunden!")

    # ------------------------------------------------------------------
    # Schritt 4: Extra-Dateien aus interim/ entfernen
    # ------------------------------------------------------------------
    interim_dir = target_dir / "interim"
    if interim_dir.exists():
        removed = []
        for pattern in EXTRA_INTERIM_PATTERNS:
            for extra_file in interim_dir.glob(pattern):
                extra_file.unlink()
                removed.append(extra_file.name)
        if removed:
            print(f"    Entfernt: {removed}")

    # ------------------------------------------------------------------
    # Schritt 5: Inhalte des negative/-Ordners löschen
    # ------------------------------------------------------------------
    negative_dir = target_dir / "negative"
    if negative_dir.exists():
        for item in negative_dir.iterdir():
            item.unlink() if item.is_file() else shutil.rmtree(item)

    # ------------------------------------------------------------------
    # Schritt 6: Berechtigungen setzen
    # ------------------------------------------------------------------
    set_permissions(target_dir)

    print(f"    OK")
    return True


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ersetzt defekte fabricad-100k Samples mit Backlog-Daten.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Beispiel:\n"
            "  sudo python3 repair_broken_samples.py\n"
            "  sudo python3 repair_broken_samples.py --dry-run"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zeigt an, was getan würde, ohne Änderungen vorzunehmen.",
    )
    args = parser.parse_args()

    if not args.dry_run and os.geteuid() != 0:
        print("FEHLER: Dieses Script muss mit sudo-Rechten ausgeführt werden.")
        print("Verwendung: sudo python3 repair_broken_samples.py")
        sys.exit(1)

    backlog_samples = get_backlog_samples()
    n_broken = len(BROKEN_IDS)
    n_backlog = len(backlog_samples)

    print(f"Defekte Samples:  {n_broken}")
    print(f"Backlog-Samples:  {n_backlog}")

    if n_backlog < n_broken:
        print(
            f"\nFEHLER: Nicht genügend Backlog-Samples ({n_backlog}) für alle "
            f"defekten IDs ({n_broken}).\n"
            f"Fehlende IDs: {BROKEN_IDS[n_backlog:]}"
        )
        sys.exit(1)

    # Zuordnung: defekte ID → Backlog-Sample (je sortiert, erste N Backlog-Samples)
    pairs = list(zip(BROKEN_IDS, backlog_samples))

    print(f"\nZuordnung ({len(pairs)} Paare):")
    for target_id, source_id in pairs:
        print(f"  {target_id}  <-  {source_id}")

    if args.dry_run:
        print("\n--- DRY RUN ---\n")
    else:
        print()
        confirm = input("Fortfahren? [y/N] ").strip().lower()
        if confirm != "y":
            print("Abgebrochen.")
            sys.exit(0)
        print()

    success = 0
    failed = []
    for target_id, source_id in pairs:
        if repair_sample(target_id, source_id, dry_run=args.dry_run):
            success += 1
        else:
            failed.append(target_id)

    prefix = "[DRY RUN] " if args.dry_run else ""
    print(f"\n{prefix}Ergebnis: {success}/{len(pairs)} Samples erfolgreich verarbeitet.")
    if failed:
        print(f"Fehlgeschlagen: {failed}")


if __name__ == "__main__":
    main()
