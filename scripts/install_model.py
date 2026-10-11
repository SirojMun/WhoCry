"""Salin model terbaik hasil tahap main ke folder model realtime (branch data-live: models/current/).

Model terbaik = run dengan loss validasi terendah di antara (model, varian) yang rerata antar-seed-nya
terendah — hanya memakai data validasi, bukan data uji.

    python scripts/install_model.py <folder hasil main> <folder tujuan>
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

FILES = ("config.json", "model.pt", "scaler.npz")


def main(main_dir, dest):
    runs = [dict(dir=p.parent, model=(c := json.loads(p.read_text()))["model"], variant=c["variant"],
                 val=pd.read_csv(p.with_name("history.csv")).val_loss.min())
            for p in Path(main_dir).glob("*/config.json") if p.with_name("model.pt").exists()]
    df = pd.DataFrame(runs)
    best_mv = df.groupby(["model", "variant"]).val.mean().idxmin()
    best = df[(df.model == best_mv[0]) & (df.variant == best_mv[1])].sort_values("val").iloc[0]
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy(best["dir"] / f, dest / f)
    print(f"terpasang: {best['dir'].name} (val_loss {best['val']:.4f}) -> {dest}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
