"""Loop pelatihan dan prediksi PyTorch yang dipakai semua model."""

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from whocry.features.build import inverse_targets
from whocry.models.nets import TRAINABLE, build_model


@dataclass
class TrainConfig:
    model: str = "gru"
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    max_epochs: int = 30
    patience: int = 5
    max_train_samples: int | None = None  # subsampel acak per epoch (mempercepat)
    seed: int = 0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iterate(data, idx, batch_size, shuffle, rng=None):
    order = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
    for i in range(0, len(order), batch_size):
        x, y = data.batch(idx[order[i:i + batch_size]])
        yield torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate_loss(model, data, idx, batch_size, dev):
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    total, n = 0.0, 0
    for x, y in iterate(data, idx, batch_size, shuffle=False):
        x, y = x.to(dev), y.to(dev)
        total += loss_fn(model(x), y).item()
        n += y.numel()
    return total / n


def make_model(cfg, data):
    hp = dict(hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout) if cfg.model in ("lstm", "gru") else {}
    return build_model(cfg.model, data.n_in, data.n_features, data.n_out, data.n_targets, **hp)


def fit(cfg, data, index, model=None, log=print):
    """Latih dengan early stopping pada data validasi. Mengembalikan model terbaik dan riwayat."""
    set_seed(cfg.seed)
    dev = device()
    model = (model or make_model(cfg, data)).to(dev)
    history = []
    if cfg.model not in TRAINABLE:
        return model, history

    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    best, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        train_idx = index["train"]
        if cfg.max_train_samples and len(train_idx) > cfg.max_train_samples:
            train_idx = train_idx[rng.choice(len(train_idx), cfg.max_train_samples, replace=False)]
        model.train()
        total, n = 0.0, 0
        for x, y in iterate(data, train_idx, cfg.batch_size, shuffle=True, rng=rng):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(x)
            n += len(x)
        val = evaluate_loss(model, data, index["val"], cfg.batch_size * 4, dev)
        history.append(dict(epoch=epoch, train_loss=total / n, val_loss=val, seconds=time.time() - t0))
        log(f"epoch {epoch:3d}  train {total / n:.4f}  val {val:.4f}  ({time.time() - t0:.0f} s)")
        if val < best - 1e-5:
            best, best_state, bad_epochs = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                log(f"early stopping pada epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, data, idx, x_scaler, batch_size=4096):
    """Prediksi dan target pada skala asli: dua array [N, n_out, n_targets]."""
    dev = device()
    model.eval().to(dev)
    preds, trues = [], []
    for x, y in iterate(data, idx, batch_size, shuffle=False):
        preds.append(model(x.to(dev)).cpu().numpy())
        trues.append(y.numpy())
    p = inverse_targets(np.concatenate(preds), x_scaler)
    y = inverse_targets(np.concatenate(trues), x_scaler)
    return y, p
