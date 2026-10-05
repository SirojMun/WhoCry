"""Tes pipeline fitur, terutama pencegahan kebocoran data."""

import numpy as np
import pandas as pd
import pytest

from whocry.features import build
from whocry.features.build import DYNAMIC_NAMES, VARIANTS, prepare, sample_index
from whocry.features.spatial import coast_distances, idw_weights

N_IN, N_OUT = 24, 24


def make_locations(n=5):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "adm2": [f"33.{i:02d}" for i in range(1, n + 1)],
        "lat": -7.0 - rng.random(n), "lon": 109.0 + rng.random(n) * 2,
        "elevasi": rng.random(n) * 500,
    })


def make_hourly(locations, start="2020-01-01", hours=24 * 40):
    """Data sintetis; temperature_2m = indeks jam (untuk melacak waktu dalam batch)."""
    times = pd.date_range(start, periods=hours, freq="h")
    rng = np.random.default_rng(1)
    frames = []
    for j, adm2 in enumerate(locations["adm2"]):
        df = pd.DataFrame({"adm2": adm2, "time": times})
        for name in DYNAMIC_NAMES + ["wind_direction_10m"]:
            if name in ("wind_u", "wind_v"):
                continue
            df[name] = rng.random(hours).astype("float32") * 10
        df["temperature_2m"] = np.arange(hours, dtype="float32") + 1000 * j
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def setup():
    loc = make_locations()
    hourly = make_hourly(loc)
    # Latih 20 hari, validasi 10 hari, uji 10 hari.
    kw = dict(train_end="2020-01-20 23:00", val_end="2020-01-30 23:00")
    return loc, hourly, kw


def raw_time(data, meta, x_or_y):
    """Kembalikan nilai temperature_2m (= indeks jam) ke skala asli."""
    sc = meta["x_scaler"]
    return x_or_y[..., 0] * sc.std[0] + sc.mean[0]


@pytest.mark.parametrize("variant", VARIANTS)
def test_shapes(setup, variant):
    loc, hourly, kw = setup
    data, index, _ = prepare(hourly, loc, variant, **kw)
    x, y = data.batch(index["train"][:8])
    assert x.shape == (8, N_IN, data.n_features)
    assert y.shape == (8, N_OUT, 4)
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_no_future_information_in_inputs(setup):
    loc, hourly, kw = setup
    data, index, meta = prepare(hourly, loc, "S0", **kw)
    idx = index["train"]
    x, y = data.batch(idx)
    tx, ty = np.rint(raw_time(data, meta, x)), np.rint(raw_time(data, meta, y))
    loc_offset = 1000 * idx[:, 1][:, None]
    # Jam terakhir masukan = t, target dimulai t+1 dan berurutan.
    assert np.array_equal(tx[:, -1] - loc_offset[:, 0], idx[:, 0])
    assert np.array_equal(ty - loc_offset, idx[:, :1] + np.arange(1, N_OUT + 1))
    assert (tx.max(axis=1) < ty.min(axis=1)).all()


def test_splits_do_not_overlap(setup):
    loc, hourly, kw = setup
    _, index, _ = prepare(hourly, loc, "S0", **kw)
    span = {k: (v[:, 0].min() - N_IN + 1, v[:, 0].max() + N_OUT) for k, v in index.items()}
    assert span["train"][1] < span["val"][0]
    assert span["val"][1] < span["test"][0]


def test_scaler_uses_train_only(setup):
    loc, hourly, kw = setup
    _, _, meta_a = prepare(hourly, loc, "S0", **kw)
    changed = hourly.copy()
    changed.loc[changed["time"] > pd.Timestamp(kw["train_end"]), "relative_humidity_2m"] = 1e6
    _, _, meta_b = prepare(changed, loc, "S0", **kw)
    np.testing.assert_allclose(meta_a["x_scaler"].mean, meta_b["x_scaler"].mean)
    np.testing.assert_allclose(meta_a["x_scaler"].std, meta_b["x_scaler"].std)


def test_windows_with_nan_are_skipped():
    X = np.ones((100, 2, 3), dtype="float32")
    X[50, 0, 1] = np.nan
    idx = sample_index(X, (0, 99))
    for t, l in idx:
        if l == 0:
            assert not (t - N_IN + 1 <= 50 <= t + N_OUT)
    assert (idx[:, 1] == 1).sum() == 100 - N_IN - N_OUT + 1


def test_s2_is_weighted_neighbor_mean(setup):
    loc, hourly, kw = setup
    data, index, _ = prepare(hourly, loc, "S2", **kw)
    idx = index["train"][:4]
    x, _ = data.batch(idx)
    F = data.X.shape[2]
    t, l = idx[0]
    expected = np.einsum("m,mf->f", data.W[l], data.X[t, :, :])
    np.testing.assert_allclose(x[0, -1, -F:], expected, rtol=1e-5)


def test_idw_weights_rows():
    loc = make_locations(6)
    W = idw_weights(loc["lat"], loc["lon"], k=3)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, rtol=1e-6)
    assert (np.diag(W) == 0).all()
    assert ((W > 0).sum(axis=1) == 3).all()


def test_coast_distance_sanity():
    # Semarang dekat pantai utara; Wonosobo jauh dari keduanya.
    d = coast_distances(np.array([-6.98, -7.36]), np.array([110.41, 109.90]))
    assert d[0, 0] < 10
    assert d[1, 0] > 40 and d[1, 1] > 40


def test_inverse_targets_roundtrip(setup):
    loc, hourly, kw = setup
    data, index, meta = prepare(hourly, loc, "S0", **kw)
    _, y = data.batch(index["val"][:16])
    out = build.inverse_targets(y, meta["x_scaler"])
    assert (out[..., 3] >= 0).all()          # curah hujan tidak negatif
    assert (out[..., 1] <= 100).all()        # kelembapan maks 100%
