"""Simpan satu snapshot prakiraan BMKG untuk semua lokasi di config/locations.csv.

    python scripts/collect_bmkg.py --out <folder>

Hasil: <folder>/bmkg/YYYY/MM/DD/HHMM.csv.gz (waktu pengambilan, UTC) dan
<folder>/bmkg/latest.csv.gz (salinan snapshot terbaru untuk dasbor).
Dijalankan tiap 3 jam oleh .github/workflows/collect-bmkg.yml.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from whocry import settings
from whocry.data.bmkg import collect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=settings.DATA_DIR)
    args = ap.parse_args()

    locations = pd.read_csv(settings.LOCATIONS_CSV, dtype={"adm2": str, "adm4_bmkg": str})
    now = datetime.now(timezone.utc)
    data, failed = collect(locations)
    for adm2, err in failed:
        print(f"gagal {adm2}: {err}", file=sys.stderr)
    if data.empty:
        sys.exit("tidak ada data yang berhasil diambil")

    data.insert(0, "fetched_at", now.strftime("%Y-%m-%d %H:%M:%S"))
    path = args.out / "bmkg" / now.strftime("%Y/%m/%d") / f"{now:%H%M}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    data.to_csv(args.out / "bmkg" / "latest.csv.gz", index=False)  # dibaca dasbor
    print(f"{path}: {len(data)} baris, {data['adm2'].nunique()} lokasi, gagal {len(failed)}")
    # Gagal sebagian masih diterima; gagal lebih dari separuh dianggap error.
    if len(failed) > len(locations) // 2:
        sys.exit(1)


if __name__ == "__main__":
    main()
