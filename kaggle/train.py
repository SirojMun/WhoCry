"""Runner eksperimen di Kaggle (GPU). Satu tahap per run, karena satu sesi Kaggle maksimal 12 jam.

Tahap (ubah STAGE lalu push: `kaggle kernels push -p kaggle`):
  search  — baseline, regresi linier S0–S3, random search hiperparameter GRU (S1)
  main    — LSTM & GRU × S0–S3 × 3 seed dengan hiperparameter BEST
  retrain — simulasi walk-forward R0–R3 pada BEST_MODEL/BEST_VARIANT

Hasil di /kaggle/working/outputs/<STAGE>/; unduh dengan
`kaggle kernels output sirojmunir/whocry-train -p outputs/kaggle`.
"""

STAGE = "main"
BEST = dict(hidden=128, layers=1, dropout=0.0, lr=1e-3)  # dari search/hp_summary.csv (2026-10-08)
BEST_MODEL, BEST_VARIANT = "gru", "S2"                    # isi dari rekap tahap main
TIME_BUDGET_H = 10.5  # berhenti memulai run baru setelah ini agar output sempat tersimpan

import glob
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path

subprocess.run("git clone -q --depth 1 https://github.com/SirojMun/akuKehujanan.git /tmp/repo",
               shell=True, check=True)
sys.path[:0] = ["/tmp/repo/src", "/tmp/repo/scripts"]  # di /tmp agar tidak ikut output

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from run_experiment import load, run  # noqa: E402
from whocry.train import TrainConfig  # noqa: E402

START = time.time()
OUT = Path("/kaggle/working/outputs") / STAGE
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "TIDAK ADA")
hourly, locations = load(glob.glob("/kaggle/input/**/hourly.parquet", recursive=True)[0])
print(f"{len(hourly):,} baris, {hourly.adm2.nunique()} lokasi")


def time_left():
    ok = time.time() - START < TIME_BUDGET_H * 3600
    if not ok:
        print("anggaran waktu habis; run berikutnya dilewati")
    return ok


def summarize_runs(root):
    """Gabungkan metrics.csv semua run menjadi satu tabel rekap."""
    rows = []
    for cfg_path in root.glob("**/config.json"):
        c = json.loads(cfg_path.read_text())
        m = pd.read_csv(cfg_path.with_name("metrics.csv"))
        rows.append(m.assign(model=c["model"], variant=c["variant"], seed=c["seed"], run_id=c["run_id"]))
    if rows:
        pd.concat(rows).to_csv(root / "all_metrics.csv", index=False)


if STAGE == "search":
    for name in ["persistence", "seasonal_naive"]:
        run(TrainConfig(model=name), "S0", hourly, locations, OUT / "baseline")
    for variant in ["S0", "S1", "S2", "S3"]:
        if time_left():
            run(TrainConfig(model="linear", max_epochs=20, max_train_samples=1_000_000),
                variant, hourly, locations, OUT / "linear")
    random.seed(0)
    space = dict(hidden=[32, 64, 128], layers=[1, 2], dropout=[0.0, 0.1, 0.2], lr=[1e-3, 3e-3])
    results = []
    for hidden, layers, dropout, lr in random.sample(list(itertools.product(*space.values())), 12):
        if not time_left():
            break
        cfg = TrainConfig(model="gru", hidden=hidden, layers=layers, dropout=dropout, lr=lr,
                          max_epochs=15, max_train_samples=300_000)
        out, _, _ = run(cfg, "S1", hourly, locations, OUT / "hp")
        hist = pd.read_csv(out / "history.csv")
        results.append(dict(hidden=hidden, layers=layers, dropout=dropout, lr=lr,
                            val_loss=hist.val_loss.min(), epochs=len(hist)))
        pd.DataFrame(results).sort_values("val_loss").to_csv(OUT / "hp_summary.csv", index=False)

elif STAGE == "main":
    for model, variant, seed in itertools.product(["lstm", "gru"], ["S0", "S1", "S2", "S3"], [0, 1, 2]):
        if time_left():
            cfg = TrainConfig(model=model, seed=seed, max_epochs=30, max_train_samples=1_000_000, **BEST)
            run(cfg, variant, hourly, locations, OUT)

elif STAGE == "retrain":
    from whocry.features.build import prepare
    from whocry.retrain import RetrainConfig, simulate
    from whocry.train import fit, make_model

    cfg = TrainConfig(model=BEST_MODEL, seed=0, max_epochs=30, max_train_samples=1_000_000, **BEST)
    data, index, meta = prepare(hourly, locations, BEST_VARIANT)
    base, _ = fit(cfg, data, index)
    OUT.mkdir(parents=True, exist_ok=True)
    for strategy in ["R0", "R1", "R2", "R3"]:
        if not time_left():
            break
        # ponytail: R1 dibatasi 3 epoch × 200 ribu sampel per minggu agar muat dalam satu sesi Kaggle.
        rcfg = RetrainConfig(strategy=strategy, full_epochs=3, full_max_samples=200_000)
        weeks = simulate(base, lambda: make_model(cfg, data), data, meta, rcfg)
        weeks.to_csv(OUT / f"retrain_{strategy}.csv", index=False)

summarize_runs(OUT)
print(f"selesai dalam {(time.time() - START) / 3600:.1f} jam")
