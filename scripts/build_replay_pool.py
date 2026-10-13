"""Bangun kumpulan replay untuk pelatihan ulang realtime: potongan acak 72 jam (semua lokasi)
dari periode latih + validasi (2017–2024). Disimpan sekali ke branch data-live: replay/pool.parquet.

    python scripts/build_replay_pool.py <file keluaran> [--blocks 300]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from whocry import settings
from whocry.realtime.predict import FETCH_VARS

BLOCK_HOURS = 72   # 1 blok = 25 sampel jendela (24 jam masukan + 24 jam target)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--blocks", type=int, default=300)
    a = ap.parse_args()
    df = pd.read_parquet(settings.DATA_DIR / "processed" / "hourly.parquet", columns=["adm2", "time", *FETCH_VARS])
    df["adm2"] = df["adm2"].astype(str)
    times = pd.date_range(settings.HISTORY_START, settings.VAL_END, freq="h")
    rng = np.random.default_rng(0)
    starts = times[rng.choice(len(times) - BLOCK_HOURS, a.blocks, replace=False)]
    keep = np.zeros(len(df), dtype=bool)
    for s in starts:
        keep |= (df["time"] >= s) & (df["time"] < s + pd.Timedelta(hours=BLOCK_HOURS))
    pool = df[keep].reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(a.out, index=False, compression="zstd")
    print(f"{a.out}: {len(pool):,} baris, {a.blocks} blok, {a.out.stat().st_size / 1e6:.1f} MB")
