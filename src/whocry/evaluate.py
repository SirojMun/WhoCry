"""Metrik evaluasi pada skala asli."""

import math

import numpy as np
import pandas as pd

from whocry.features.build import TARGET_NAMES

REPORT_HORIZONS = (1, 3, 6, 12, 24)
RAIN_THRESHOLDS = (0.1, 1.0)  # mm/jam
PRECIP = TARGET_NAMES.index("precipitation")


def regression_metrics(y, p):
    """y, p: [N, H, K]. Keluaran DataFrame (horizon, target) -> MAE, RMSE, R²."""
    err = p - y
    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    ss_res = (err ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)
    rows = []
    for h in range(y.shape[1]):
        for k, name in enumerate(TARGET_NAMES):
            rows.append(dict(horizon=h + 1, target=name, mae=mae[h, k], rmse=rmse[h, k], r2=r2[h, k]))
    return pd.DataFrame(rows)


def rain_detection(y, p, thresholds=RAIN_THRESHOLDS):
    """POD, FAR, CSI curah hujan per horizon dan ambang."""
    rows = []
    for thr in thresholds:
        obs, fc = y[..., PRECIP] >= thr, p[..., PRECIP] >= thr
        hits = (obs & fc).sum(axis=0)
        misses = (obs & ~fc).sum(axis=0)
        false_alarms = (~obs & fc).sum(axis=0)
        for h in range(y.shape[1]):
            a, b, c = hits[h], misses[h], false_alarms[h]
            rows.append(dict(
                horizon=h + 1, threshold=thr,
                pod=a / (a + b) if a + b else np.nan,
                far=c / (a + c) if a + c else np.nan,
                csi=a / (a + b + c) if a + b + c else np.nan,
            ))
    return pd.DataFrame(rows)


def skill_score(metrics, reference, metric="mae"):
    """1 - metrik_model / metrik_pembanding untuk setiap (horizon, target)."""
    m = metrics.set_index(["horizon", "target"])[metric]
    r = reference.set_index(["horizon", "target"])[metric]
    return (1 - m / r).rename(f"skill_{metric}").reset_index()


def diebold_mariano(e1, e2, h=1, power=1):
    """Uji Diebold–Mariano dua sisi (koreksi Harvey dkk. 1997).

    e1, e2: galat dua model pada sampel yang sama, berurutan menurut waktu.
    Statistik negatif berarti model 1 lebih akurat.
    """
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    d_mean = d.mean()
    gamma = [np.mean((d[k:] - d_mean) * (d[: n - k] - d_mean)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    stat = d_mean / math.sqrt(var) if var > 0 else np.nan
    stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    # Distribusi t dengan n-1 derajat bebas ~ normal untuk n besar.
    p = math.erfc(abs(stat) / math.sqrt(2))
    return stat, p


def summarize(metrics, horizons=REPORT_HORIZONS):
    """Tabel ringkas: baris = target, kolom = MAE per horizon pelaporan."""
    sub = metrics[metrics["horizon"].isin(horizons)]
    return sub.pivot(index="target", columns="horizon", values="mae").reindex(TARGET_NAMES).astype(float)
