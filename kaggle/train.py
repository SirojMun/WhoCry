"""Runner eksperimen di Kaggle (GPU). Push: `kaggle kernels push -p kaggle`.

Tahap (dijalankan berurutan sesuai STAGES, dibatasi TIME_BUDGET_H per sesi):
  search  — baseline, regresi linier S0–S4, random search hiperparameter GRU (S1)
  main    — LSTM & GRU × S0–S4 × 3 seed dengan hiperparameter terbaik dari search
  retrain — simulasi walk-forward R0–R3 pada model & varian dengan loss validasi terendah

Pemilihan hiperparameter dan model hanya memakai data validasi (bukan data uji).
Hasil di /kaggle/working/outputs/<tahap>/; unduh dengan
`kaggle kernels output sirojmunir/whocry-train -p outputs/kaggle`.
"""

STAGES = ["retrain"]   # search & main 40 lokasi selesai 2026-10-13 (kernel v3)
VARIANTS = ["S0", "S1", "S2", "S3", "S4"]
RETRAIN_MODEL = ("gru", "S4")              # None = pilih otomatis dari loss validasi tahap main
RETRAIN_STRATEGIES = ["R4"]   # ulang R4 setelah perbaikan tumpang tindih target latih–gerbang (R0 deterministik)
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
ROOT = Path("/kaggle/working/outputs")
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


def best_hparams():
    hp = ROOT / "search" / "hp_summary.csv"
    if not hp.exists():  # fallback: hasil search 35 lokasi (2026-10-08)
        return dict(hidden=128, layers=1, dropout=0.0, lr=1e-3)
    r = pd.read_csv(hp).iloc[0]
    return dict(hidden=int(r.hidden), layers=int(r.layers), dropout=float(r.dropout), lr=float(r.lr))


def best_model_variant():
    """Model & varian dengan rerata (antar-seed) loss validasi terendah di tahap main."""
    rows = [dict(model=(c := json.loads(p.read_text()))["model"], variant=c["variant"],
                 val=pd.read_csv(p.with_name("history.csv")).val_loss.min())
            for p in (ROOT / "main").glob("*/config.json")]
    if not rows:
        return "gru", "S0"
    s = pd.DataFrame(rows).groupby(["model", "variant"]).val.mean().sort_values()
    s.to_csv(ROOT / "main" / "val_ranking.csv")
    return s.index[0]


for stage in STAGES:
    out = ROOT / stage
    print(f"===== tahap {stage} =====")
    if stage == "search":
        for name in ["persistence", "seasonal_naive"]:
            run(TrainConfig(model=name), "S0", hourly, locations, out / "baseline")
        for variant in VARIANTS:
            if time_left():
                run(TrainConfig(model="linear", max_epochs=20, max_train_samples=1_000_000),
                    variant, hourly, locations, out / "linear")
        random.seed(0)
        space = dict(hidden=[32, 64, 128], layers=[1, 2], dropout=[0.0, 0.1, 0.2], lr=[1e-3, 3e-3])
        results = []
        for hidden, layers, dropout, lr in random.sample(list(itertools.product(*space.values())), 12):
            if not time_left():
                break
            cfg = TrainConfig(model="gru", hidden=hidden, layers=layers, dropout=dropout, lr=lr,
                              max_epochs=15, max_train_samples=300_000)
            run_dir, _, _ = run(cfg, "S1", hourly, locations, out / "hp")
            hist = pd.read_csv(run_dir / "history.csv")
            results.append(dict(hidden=hidden, layers=layers, dropout=dropout, lr=lr,
                                val_loss=hist.val_loss.min(), epochs=len(hist)))
            pd.DataFrame(results).sort_values("val_loss").to_csv(out / "hp_summary.csv", index=False)

    elif stage == "main":
        best = best_hparams()
        print("hiperparameter:", best)
        for model, variant, seed in itertools.product(["lstm", "gru"], VARIANTS, [0, 1, 2]):
            if time_left():
                cfg = TrainConfig(model=model, seed=seed, max_epochs=30, max_train_samples=1_000_000, **best)
                run(cfg, variant, hourly, locations, out)

    elif stage == "retrain":
        from whocry.features.build import prepare
        from whocry.retrain import RetrainConfig, simulate
        from whocry.train import fit, make_model

        model_name, variant = RETRAIN_MODEL or best_model_variant()
        print("model untuk simulasi pelatihan ulang:", model_name, variant)
        cfg = TrainConfig(model=model_name, seed=0, max_epochs=30, max_train_samples=1_000_000, **best_hparams())
        data, index, meta = prepare(hourly, locations, variant)
        base, _ = fit(cfg, data, index)
        out.mkdir(parents=True, exist_ok=True)
        for strategy in RETRAIN_STRATEGIES:
            if not time_left():
                break
            # ponytail: R1 dibatasi 3 epoch × 200 ribu sampel per minggu agar muat dalam satu sesi Kaggle.
            rcfg = RetrainConfig(strategy=strategy, full_epochs=3, full_max_samples=200_000)
            weeks = simulate(base, lambda: make_model(cfg, data), data, meta, rcfg)
            weeks.to_csv(out / f"retrain_{strategy}.csv", index=False)
            (out / "model.txt").write_text(f"{model_name} {variant}\n")

    summarize_runs(out)

print(f"selesai dalam {(time.time() - START) / 3600:.1f} jam")
