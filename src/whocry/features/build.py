"""Pembentukan tensor fitur, normalisasi, pembagian data, dan pengambilan batch.

Data disusun sebagai array padat X[T, L, F] (waktu × lokasi × fitur dinamis).
Jendela tidak disalin ke memori; batch dibentuk saat dibutuhkan lewat indexing.

Sebuah sampel (t, l) memakai masukan pada jam t-23..t dan target pada jam t+1..t+24.
Sampel masuk ke suatu bagian (latih/validasi/uji) hanya jika seluruh jam masukan
dan targetnya berada di dalam rentang bagian itu, sehingga tidak ada tumpang tindih
antarbagian.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from whocry import settings
from whocry.features.spatial import STATIC_NAMES, idw_weights, static_features

TARGET_NAMES = ["temperature_2m", "relative_humidity_2m", "cloud_cover", "precipitation"]
TARGET_RANGES = {"relative_humidity_2m": (0, 100), "cloud_cover": (0, 100), "precipitation": (0, None)}
DYNAMIC_NAMES = TARGET_NAMES + [
    "dew_point_2m", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "pressure_msl", "surface_pressure", "wind_speed_10m", "wind_gusts_10m",
    "shortwave_radiation", "wind_u", "wind_v",
]
TIME_NAMES = ["jam_sin", "jam_cos", "hari_sin", "hari_cos"]
LOG_VARS = {"precipitation"}
VARIANTS = ("S0", "S1", "S2", "S3")
WIB_OFFSET_H = 7


def add_derived(df):
    """Komponen angin u/v (arah datang angin -> vektor) menggantikan derajat mentah."""
    rad = np.deg2rad(df["wind_direction_10m"].astype("float64"))
    df["wind_u"] = (-df["wind_speed_10m"] * np.sin(rad)).astype("float32")
    df["wind_v"] = (-df["wind_speed_10m"] * np.cos(rad)).astype("float32")
    return df


def to_array(hourly, locations, max_gap_h=3):
    """DataFrame panjang (adm2, time, var...) -> X[T, L, F] dengan grid waktu per jam yang lengkap."""
    hourly = add_derived(hourly.copy())
    times = pd.date_range(hourly["time"].min(), hourly["time"].max(), freq="h")
    order = locations["adm2"].tolist()
    X = np.full((len(times), len(order), len(DYNAMIC_NAMES)), np.nan, dtype="float32")
    for j, adm2 in enumerate(order):
        part = (hourly[hourly["adm2"] == adm2].set_index("time")[DYNAMIC_NAMES]
                .reindex(times).interpolate(limit=max_gap_h, limit_area="inside"))
        X[:, j, :] = part.to_numpy(dtype="float32")
    for k, name in enumerate(DYNAMIC_NAMES):
        if name in LOG_VARS:
            X[:, :, k] = np.log1p(np.clip(X[:, :, k], 0, None))
    return X, times


def time_features(times):
    local = times + pd.Timedelta(hours=WIB_OFFSET_H)
    hour = 2 * np.pi * local.hour / 24
    doy = 2 * np.pi * (local.dayofyear - 1) / 365.25
    return np.column_stack([np.sin(hour), np.cos(hour), np.sin(doy), np.cos(doy)]).astype("float32")


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, a, axis):
        mean = np.nanmean(a, axis=axis)
        std = np.nanstd(a, axis=axis)
        return cls(mean.astype("float32"), np.where(std < 1e-6, 1.0, std).astype("float32"))

    def transform(self, a):
        return ((a - self.mean) / self.std).astype("float32")

    def inverse(self, a):
        return a * self.std + self.mean


def split_bounds(times, train_end=settings.TRAIN_END, val_end=settings.VAL_END):
    """Indeks [awal, akhir] (inklusif) untuk setiap bagian."""
    train_end = times.get_indexer([pd.Timestamp(train_end)])[0]
    val_end = times.get_indexer([pd.Timestamp(val_end)])[0]
    if train_end < 0 or val_end < 0:
        raise ValueError("TRAIN_END/VAL_END berada di luar rentang data")
    return {"train": (0, train_end), "val": (train_end + 1, val_end), "test": (val_end + 1, len(times) - 1)}


def sample_index(X, bounds, n_in=settings.INPUT_HOURS, n_out=settings.HORIZON_HOURS, stride=1):
    """Pasangan (t, l) yang valid: jendela lengkap di dalam bagian dan tanpa NaN."""
    lo, hi = bounds
    ts = np.arange(lo + n_in - 1, hi - n_out + 1, stride)
    finite = np.isfinite(X).all(axis=2)  # [T, L]
    # Jumlah kumulatif untuk mengecek NaN di seluruh jendela secara vektor.
    bad = np.vstack([np.zeros((1, X.shape[1]), dtype=int), np.cumsum(~finite, axis=0)])
    start, stop = ts - n_in + 1, ts + n_out + 1
    ok = (bad[stop] - bad[start]) == 0  # [len(ts), L]
    t_idx, l_idx = np.nonzero(ok)
    return np.column_stack([ts[t_idx], l_idx]).astype("int64")


@dataclass
class WindowData:
    """Semua array yang dibutuhkan untuk membentuk batch, untuk satu varian spasial."""

    X: np.ndarray            # [T, L, F] ternormalisasi
    T_feat: np.ndarray       # [T, 4]
    static: np.ndarray       # [L, S] ternormalisasi
    W: np.ndarray            # [L, L] bobot IDW
    variant: str
    n_in: int = settings.INPUT_HOURS
    n_out: int = settings.HORIZON_HOURS
    n_targets: int = len(TARGET_NAMES)
    _offsets_in: np.ndarray = field(init=False, repr=False)
    _offsets_out: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"varian tidak dikenal: {self.variant}")
        self._offsets_in = np.arange(-self.n_in + 1, 1)
        self._offsets_out = np.arange(1, self.n_out + 1)

    @property
    def n_locations(self):
        return self.X.shape[1]

    @property
    def n_features(self):
        f = self.X.shape[2] + self.T_feat.shape[1]
        if self.variant in ("S1", "S2", "S3"):
            f += self.static.shape[1]
        if self.variant == "S2":
            f += self.X.shape[2]
        if self.variant == "S3":
            f += self.X.shape[1] * self.X.shape[2] + self.X.shape[1]
        return f

    def batch(self, idx):
        """idx: [B, 2] berisi (t, l). Keluaran x [B, n_in, F] dan y [B, n_out, n_targets]."""
        t, l = idx[:, 0], idx[:, 1]
        tin = t[:, None] + self._offsets_in             # [B, n_in]
        tout = t[:, None] + self._offsets_out           # [B, n_out]
        parts = [self.X[tin, l[:, None]], self.T_feat[tin]]
        if self.variant in ("S1", "S2", "S3"):
            parts.append(np.repeat(self.static[l][:, None, :], self.n_in, axis=1))
        if self.variant == "S2":
            # Rata-rata berbobot tetangga: sum_m W[l, m] * X[t, m, :]
            parts.append(np.einsum("bm,btmf->btf", self.W[l], self.X[tin]))
        if self.variant == "S3":
            B = len(idx)
            parts.append(self.X[tin].reshape(B, self.n_in, -1))
            onehot = np.zeros((B, self.n_in, self.n_locations), dtype="float32")
            onehot[np.arange(B), :, l] = 1.0
            parts.append(onehot)
        x = np.concatenate(parts, axis=2).astype("float32")
        y = self.X[tout, l[:, None], : self.n_targets]
        return x, y


def prepare(hourly, locations, variant, stride=1,
            train_end=settings.TRAIN_END, val_end=settings.VAL_END):
    """Pipeline lengkap: array -> scaler dari data latih -> indeks sampel per bagian."""
    X, times = to_array(hourly, locations)
    bounds = split_bounds(times, train_end, val_end)
    lo, hi = bounds["train"]
    x_scaler = Scaler.fit(X[lo:hi + 1].reshape(-1, X.shape[2]), axis=0)
    static = static_features(locations)
    s_scaler = Scaler.fit(static, axis=0)
    data = WindowData(
        X=x_scaler.transform(X),
        T_feat=time_features(times),
        static=s_scaler.transform(static),
        W=idw_weights(locations["lat"], locations["lon"]),
        variant=variant,
    )
    index = {name: sample_index(X, b, stride=stride) for name, b in bounds.items()}
    meta = dict(times=times, bounds=bounds, x_scaler=x_scaler, s_scaler=s_scaler,
                dynamic_names=DYNAMIC_NAMES, static_names=STATIC_NAMES)
    return data, index, meta


def inverse_targets(y, x_scaler):
    """Kembalikan target ternormalisasi ke satuan asli (curah hujan: log1p -> mm)."""
    k = len(TARGET_NAMES)
    out = y * x_scaler.std[:k] + x_scaler.mean[:k]
    for j, name in enumerate(TARGET_NAMES):
        if name in LOG_VARS:
            out[..., j] = np.expm1(out[..., j])
        if name in TARGET_RANGES:
            out[..., j] = out[..., j].clip(*TARGET_RANGES[name])
    return out
