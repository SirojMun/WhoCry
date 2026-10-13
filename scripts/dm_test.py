"""Uji Diebold–Mariano antarmodel pada data uji (2025-01 s.d. akhir data).

Galat absolut dirata-ratakan antarlokasi per jam terbit, sehingga terbentuk deret waktu galat
untuk setiap (model, target, horizon); uji DM memakai koreksi HAC sesuai horizon.

    python scripts/dm_test.py <folder hasil main> <folder hasil search> [--out reports/dm_tests.csv]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiment import load
from whocry.evaluate import diebold_mariano
from whocry.features.build import TARGET_NAMES, prepare
from whocry.models.nets import Persistence, SeasonalNaive
from whocry.realtime.predict import load_artifact
from whocry.train import predict

HORIZONS = (1, 3, 6, 24)
PAIRS = [  # (model A, model B): statistik negatif = A lebih akurat
    ("gru_S4", "gru_S0"), ("linear_S4", "linear_S0"), ("gru_S4", "linear_S4"),
    ("gru_S0", "linear_S0"), ("gru_S4", "lstm_S4"), ("gru_S4", "naif_terbaik"),
]


def run_dir(root, name, seed=0):
    """Folder run untuk 'model_varian' (seed 0)."""
    for p in Path(root).glob("**/config.json"):
        c = json.loads(p.read_text())
        if f"{c['model']}_{c['variant']}" == name and c["seed"] == seed and p.with_name("model.pt").exists():
            return p.parent
    raise FileNotFoundError(name)


def series_errors(y, p, idx):
    """[N, H, K] -> dict (target, horizon) -> deret rerata galat absolut per jam terbit."""
    t = idx[:, 0]
    out = {}
    for k, name in enumerate(TARGET_NAMES):
        for h in HORIZONS:
            e = pd.Series(np.abs(p[:, h - 1, k] - y[:, h - 1, k])).groupby(t).mean()
            out[(name, h)] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main_dir")
    ap.add_argument("search_dir")
    ap.add_argument("--out", type=Path, default=Path("reports/dm_tests.csv"))
    a = ap.parse_args()
    hourly, locations = load()
    errors = {}
    for variant, names in (("S0", ["gru_S0", "linear_S0", "naif"]), ("S4", ["gru_S4", "linear_S4", "lstm_S4"])):
        data, index, meta = prepare(hourly, locations, variant)
        idx = index["test"]
        for name in names:
            if name == "naif":   # baseline naif terbaik (MAE terendah) per (target, horizon)
                y, p1 = predict(Persistence(data.n_out, data.n_targets), data, idx, meta["x_scaler"])
                _, p2 = predict(SeasonalNaive(data.n_out, data.n_targets), data, idx, meta["x_scaler"])
                e1, e2 = series_errors(y, p1, idx), series_errors(y, p2, idx)
                errors["naif_terbaik"] = {k: e1[k] if e1[k].mean() <= e2[k].mean() else e2[k] for k in e1}
                continue
            art = load_artifact(run_dir(a.main_dir if not name.startswith("linear") else a.search_dir, name),
                                locations)
            y, p = predict(art["model"], data, idx, meta["x_scaler"])
            errors[name] = series_errors(y, p, idx)
            print(f"{name}: {len(idx):,} sampel uji")
        del data, index, meta

    rows = []
    for a_name, b_name in PAIRS:
        for (target, h), ea in errors[a_name].items():
            eb = errors[b_name][(target, h)]
            stat, pval = diebold_mariano(ea.to_numpy(), eb.to_numpy(), h=h)
            rows.append(dict(A=a_name, B=b_name, target=target, horizon=h, mae_A=ea.mean(), mae_B=eb.mean(),
                             dm_stat=stat, p_value=pval,
                             kesimpulan="A lebih baik" if pval < 0.05 and stat < 0 else
                             "B lebih baik" if pval < 0.05 else "tidak berbeda nyata"))
    res = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
