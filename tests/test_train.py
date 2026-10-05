"""Uji asap (smoke test): semua model bisa dilatih, memprediksi, dan dievaluasi."""

import numpy as np
import pytest

from test_features import make_hourly, make_locations
from whocry.evaluate import diebold_mariano, rain_detection, regression_metrics, skill_score
from whocry.features.build import prepare
from whocry.train import TrainConfig, fit, predict

KW = dict(train_end="2020-01-20 23:00", val_end="2020-01-30 23:00")


@pytest.fixture(scope="module")
def prepared():
    loc = make_locations(4)
    return prepare(make_hourly(loc), loc, "S2", **KW)


@pytest.mark.parametrize("name", ["persistence", "seasonal_naive", "linear", "lstm", "gru"])
def test_model_end_to_end(prepared, name):
    data, index, meta = prepared
    cfg = TrainConfig(model=name, hidden=8, max_epochs=2, batch_size=64)
    model, history = fit(cfg, data, index, log=lambda *_: None)
    y, p = predict(model, data, index["test"], meta["x_scaler"])
    assert y.shape == p.shape == (len(index["test"]), 24, 4)
    m = regression_metrics(y, p)
    assert len(m) == 24 * 4 and np.isfinite(m["mae"]).all()
    assert len(rain_detection(y, p)) == 24 * 2
    if name in ("linear", "lstm", "gru"):
        assert len(history) >= 1


def test_persistence_repeats_last_value(prepared):
    data, index, meta = prepared
    model, _ = fit(TrainConfig(model="persistence"), data, index)
    y, p = predict(model, data, index["val"][:5], meta["x_scaler"])
    assert np.allclose(p[:, 0], p[:, -1])


def test_skill_score_and_dm():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(200, 24, 4))
    good, bad = y + rng.normal(0, 0.1, y.shape), y + rng.normal(0, 1.0, y.shape)
    s = skill_score(regression_metrics(y, good), regression_metrics(y, bad))
    assert (s["skill_mae"] > 0.5).all()
    stat, p = diebold_mariano((good - y)[:, 0, 0], (bad - y)[:, 0, 0])
    assert stat < 0 and p < 0.01


@pytest.mark.parametrize("strategy", ["R0", "R1", "R2", "R3"])
def test_retrain_simulation(prepared, strategy):
    from whocry.retrain import RetrainConfig, simulate
    from whocry.train import make_model

    data, index, meta = prepared
    cfg = TrainConfig(model="gru", hidden=8, max_epochs=1, batch_size=64)
    model, _ = fit(cfg, data, index, log=lambda *_: None)
    rcfg = RetrainConfig(strategy=strategy, finetune_epochs=1, full_epochs=1, batch_size=64,
                         trigger_ratio=0.0)  # R3 selalu terpicu agar jalurnya teruji
    weeks = simulate(model, lambda: make_model(cfg, data), data, meta, rcfg, log=lambda *_: None)
    assert len(weeks) >= 1
    assert {"mae_temperature_2m", "updated", "update_seconds"} <= set(weeks.columns)
    assert weeks["updated"].all() == (strategy != "R0")
