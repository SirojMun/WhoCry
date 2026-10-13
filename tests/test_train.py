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


@pytest.mark.parametrize("strategy", ["R0", "R1", "R2", "R3", "R2R", "R4"])
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
    if strategy == "R4":   # gerbang: boleh menolak challenger, tapi kolomnya harus boolean
        assert weeks["updated"].dtype == bool
    else:
        assert weeks["updated"].all() == (strategy != "R0")


def test_gate_rejects_worse_challenger(prepared, monkeypatch):
    """R4 tidak boleh memakai challenger yang lebih buruk; R2R (tanpa gerbang) selalu memakainya."""
    import torch
    from whocry import retrain
    from whocry.retrain import RetrainConfig, simulate

    data, index, meta = prepared
    champion, _ = fit(TrainConfig(model="gru", hidden=8, max_epochs=1, batch_size=64), data, index,
                      log=lambda *_: None)

    class Broken(torch.nn.Module):          # challenger buruk: prediksi konstan 10 sd dari rerata
        def forward(self, x):
            return torch.full((len(x), data.n_out, data.n_targets), 10.0)

    monkeypatch.setattr(retrain, "_train_epochs", lambda *a, **k: Broken())
    gated = simulate(champion, None, data, meta, RetrainConfig(strategy="R4", batch_size=64), log=lambda *_: None)
    ungated = simulate(champion, None, data, meta, RetrainConfig(strategy="R2R", batch_size=64), log=lambda *_: None)
    assert not gated["updated"].any()
    assert ungated["updated"].all()
