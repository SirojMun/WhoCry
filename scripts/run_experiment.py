"""Jalankan satu eksperimen: model × varian spasial × seed, lalu evaluasi pada data uji.

    python scripts/run_experiment.py --model gru --variant S2 --seed 0
    python scripts/run_experiment.py --model persistence --variant S0

Hasil disimpan di outputs/<run_id>/: config.json, history.csv, metrics.csv,
rain.csv, model.pt (untuk model yang dilatih).
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from whocry import settings
from whocry.evaluate import rain_detection, regression_metrics, summarize
from whocry.features.build import prepare
from whocry.train import TrainConfig, fit, predict


def load(data_path=None):
    hourly = pd.read_parquet(data_path or settings.DATA_DIR / "processed" / "hourly.parquet")
    hourly["adm2"] = hourly["adm2"].astype(str)
    locations = pd.read_csv(settings.LOCATIONS_CSV, dtype={"adm2": str, "adm4_bmkg": str})
    return hourly, locations


def run(cfg, variant, hourly, locations, out_dir, stride=1, max_test_samples=None, log=print):
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.model}_{variant}_s{cfg.seed}"
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    data, index, meta = prepare(hourly, locations, variant, stride=stride)
    log(f"[{run_id}] fitur={data.n_features}  latih={len(index['train']):,}  "
        f"val={len(index['val']):,}  uji={len(index['test']):,}")

    model, history = fit(cfg, data, index, log=log)
    test_idx = index["test"]
    if max_test_samples and len(test_idx) > max_test_samples:
        test_idx = test_idx[:: len(test_idx) // max_test_samples + 1]
    y, p = predict(model, data, test_idx, meta["x_scaler"])
    metrics = regression_metrics(y, p)
    rain = rain_detection(y, p)

    (out / "config.json").write_text(json.dumps(
        dict(run_id=run_id, variant=variant, stride=stride, n_features=data.n_features,
             n_params=sum(p.numel() for p in model.parameters()), **asdict(cfg)), indent=1))
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    rain.to_csv(out / "rain.csv", index=False)
    if history:
        torch.save(model.state_dict(), out / "model.pt")
        # Scaler ikut disimpan agar model bisa dipakai di pipeline realtime.
        np.savez(out / "scaler.npz", x_mean=meta["x_scaler"].mean, x_std=meta["x_scaler"].std,
                 s_mean=meta["s_scaler"].mean, s_std=meta["s_scaler"].std)
    log(summarize(metrics).round(3).to_string())
    return out, metrics, rain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru",
                    choices=["persistence", "seasonal_naive", "linear", "lstm", "gru"])
    ap.add_argument("--variant", default="S1", choices=["S0", "S1", "S2", "S3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1, help="jarak jam antar-sampel (>1 untuk uji cepat)")
    ap.add_argument("--max-test-samples", type=int, default=None)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=settings.ROOT / "outputs")
    a = ap.parse_args()
    cfg = TrainConfig(model=a.model, hidden=a.hidden, layers=a.layers, dropout=a.dropout, lr=a.lr,
                      batch_size=a.batch_size, max_epochs=a.max_epochs,
                      max_train_samples=a.max_train_samples, seed=a.seed)
    hourly, locations = load(a.data)
    run(cfg, a.variant, hourly, locations, a.out, stride=a.stride, max_test_samples=a.max_test_samples)


if __name__ == "__main__":
    main()
