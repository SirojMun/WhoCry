"""Pelatihan ulang mingguan champion–challenger (dijalankan oleh .github/workflows/retrain.yml).

    python scripts/retrain_live.py --live <folder data-live> [--force]
"""

import argparse
from pathlib import Path

from whocry.realtime.retrain_live import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=Path, required=True)
    ap.add_argument("--force", action="store_true", help="jalankan walau belum seminggu sejak pelatihan terakhir")
    a = ap.parse_args()
    run(a.live, force=a.force)
