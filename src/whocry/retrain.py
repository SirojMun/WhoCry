"""Simulasi walk-forward strategi pelatihan ulang R0–R3 pada periode uji.

Setiap minggu: (1) model saat ini memprediksi sampel minggu itu, galatnya dicatat;
(2) model diperbarui sesuai strategi, hanya memakai sampel yang targetnya sudah
diketahui pada akhir minggu itu (t + horizon <= akhir minggu).
"""

import copy
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from whocry.features.build import TARGET_NAMES, inverse_targets, sample_index
from whocry.train import device, iterate, predict

STRATEGIES = ("R0", "R1", "R2", "R3")
HOURS_PER_WEEK = 24 * 7


@dataclass
class RetrainConfig:
    strategy: str = "R2"
    finetune_weeks: int = 8       # R2/R3: jendela data terbaru
    finetune_epochs: int = 2
    finetune_lr: float = 3e-4
    full_epochs: int = 10         # R1: jumlah epoch latih ulang penuh
    full_max_samples: int = 500_000
    trigger_ratio: float = 1.1    # R3: ambang galat relatif terhadap galat validasi
    batch_size: int = 1024
    seed: int = 0


def _train_epochs(model, data, idx, epochs, lr, batch_size, rng, weight_decay=1e-4):
    dev = device()
    model.train().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for x, y in iterate(data, idx, batch_size, shuffle=True, rng=rng):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def _norm_mae(model, data, idx, batch_size=4096):
    dev = device()
    model.eval().to(dev)
    total, n = 0.0, 0
    for x, y in iterate(data, idx, batch_size, shuffle=False):
        total += (model(x.to(dev)).cpu() - y).abs().sum().item()
        n += y.numel()
    return total / max(n, 1)


def simulate(base_model, model_factory, data, meta, rcfg, log=print):
    """Jalankan satu strategi. model_factory() membuat model baru (untuk R1).

    Keluaran: DataFrame per minggu berisi MAE per target (skala asli), apakah model
    diperbarui, dan durasi pembaruan.
    """
    if rcfg.strategy not in STRATEGIES:
        raise ValueError(rcfg.strategy)
    rng = np.random.default_rng(rcfg.seed)
    model = copy.deepcopy(base_model)
    T = data.X.shape[0]
    all_idx = sample_index(data.X, (0, T - 1), n_in=data.n_in, n_out=data.n_out)
    test_lo = meta["bounds"]["test"][0]
    val_lo, val_hi = meta["bounds"]["val"]
    val_idx = all_idx[(all_idx[:, 0] - data.n_in + 1 >= val_lo) & (all_idx[:, 0] + data.n_out <= val_hi)]
    ref_mae = _norm_mae(model, data, val_idx)

    rows = []
    week_starts = range(test_lo + data.n_in - 1, T - data.n_out, HOURS_PER_WEEK)
    for w, ws in enumerate(week_starts):
        we = ws + HOURS_PER_WEEK  # eksklusif
        week_idx = all_idx[(all_idx[:, 0] >= ws) & (all_idx[:, 0] < we)]
        if len(week_idx) == 0:
            continue
        y, p = predict(model, data, week_idx, meta["x_scaler"])
        mae = np.abs(p - y).mean(axis=(0, 1))
        week_norm_mae = _norm_mae(model, data, week_idx)

        # Sampel yang targetnya sudah teramati di akhir minggu ini.
        known = all_idx[all_idx[:, 0] + data.n_out < we]
        recent = known[known[:, 0] >= we - rcfg.finetune_weeks * HOURS_PER_WEEK]
        updated, t0 = False, time.time()
        if rcfg.strategy == "R1":
            sub = known if len(known) <= rcfg.full_max_samples else \
                known[rng.choice(len(known), rcfg.full_max_samples, replace=False)]
            model = _train_epochs(model_factory(), data, sub, rcfg.full_epochs, 1e-3, rcfg.batch_size, rng)
            updated = True
        elif rcfg.strategy == "R2" or (
                rcfg.strategy == "R3" and week_norm_mae > rcfg.trigger_ratio * ref_mae):
            model = _train_epochs(model, data, recent, rcfg.finetune_epochs, rcfg.finetune_lr,
                                  rcfg.batch_size, rng)
            updated = True

        row = dict(week=w, start=meta["times"][ws], n=len(week_idx), updated=updated,
                   update_seconds=time.time() - t0, norm_mae=week_norm_mae)
        row.update({f"mae_{name}": mae[k] for k, name in enumerate(TARGET_NAMES)})
        rows.append(row)
        log(f"{rcfg.strategy} minggu {w:3d} {row['start']:%Y-%m-%d}  norm_mae {week_norm_mae:.4f}"
            f"{'  (diperbarui)' if updated else ''}")
    return pd.DataFrame(rows)
