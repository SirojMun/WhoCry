import json

import numpy as np
import pandas as pd
import torch

from test_features import make_hourly, make_locations
from whocry.features.build import prepare
from whocry.realtime import alerts
from whocry.realtime.predict import forecast, load_artifact
from whocry.train import TrainConfig, fit

KW = dict(train_end="2020-01-20 23:00", val_end="2020-01-30 23:00")


def make_pred(values, var="precipitation"):
    t0 = pd.Timestamp("2026-10-05 17:00")
    rows = []
    for h, v in enumerate(values, 1):
        rows.append(dict(adm2="33.01", horizon=h, valid_time=t0 + pd.Timedelta(hours=h), model="m",
                         temperature_2m=30.0, relative_humidity_2m=80.0, cloud_cover=50.0, precipitation=0.0))
        rows[-1][var] = v
    return pd.DataFrame(rows)


def test_find_events_takes_extreme_once_per_day():
    loc = pd.DataFrame({"adm2": ["33.01"], "nama": ["Kabupaten Cilacap"]})
    pred = make_pred([0, 12, 25, 3] + [0] * 20)
    ev = alerts.find_events(pred, loc)
    assert len(ev) == 1 and ev[0]["value"] == 25 and ev[0]["label"] == "Hujan lebat"


def test_low_humidity_threshold():
    loc = pd.DataFrame({"adm2": ["33.01"], "nama": ["X"]})
    ev = alerts.find_events(make_pred([60, 35, 38] + [70] * 21, var="relative_humidity_2m"), loc)
    assert len(ev) == 1 and ev[0]["value"] == 35


def test_process_without_credentials_does_not_mark_sent(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    loc = pd.DataFrame({"adm2": ["33.01"], "nama": ["X"]})
    state = tmp_path / "state.json"
    new = alerts.process(make_pred([20] * 24), loc, state)
    assert len(new) == 2   # 24 jam dari 18:00 UTC melintasi dua tanggal WIB
    assert json.loads(state.read_text()) == {}   # tidak terkirim -> dicoba lagi nanti


def test_forecast_with_trained_artifact(tmp_path):
    loc = make_locations(4)
    hourly = make_hourly(loc)
    data, index, meta = prepare(hourly, loc, "S2", **KW)
    cfg = TrainConfig(model="gru", hidden=8, max_epochs=1, batch_size=64)
    model, _ = fit(cfg, data, index, log=lambda *_: None)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    np.savez(tmp_path / "scaler.npz", x_mean=meta["x_scaler"].mean, x_std=meta["x_scaler"].std,
             s_mean=meta["s_scaler"].mean, s_std=meta["s_scaler"].std)
    (tmp_path / "config.json").write_text(json.dumps(dict(
        model="gru", variant="S2", n_features=data.n_features, hidden=8, layers=1, dropout=0.1)))

    artifact = load_artifact(tmp_path, loc)
    recent = hourly[hourly["time"] >= hourly["time"].max() - pd.Timedelta(hours=47)]
    pred = forecast(recent, loc, artifact)
    assert len(pred) == 4 * 24
    assert (pred["model"] == "gru_S2").all()
    assert pred["valid_time"].min() == recent["time"].max() + pd.Timedelta(hours=1)
    assert pred[["temperature_2m", "precipitation"]].notna().all().all()


def test_forecast_persistence_fallback():
    loc = make_locations(3)
    hourly = make_hourly(loc)
    pred = forecast(hourly, loc, None)
    last = hourly[hourly["time"] == hourly["time"].max()].set_index("adm2")["temperature_2m"]
    got = pred[pred["horizon"] == 24].set_index("adm2")["temperature_2m"]
    np.testing.assert_allclose(got.loc[last.index], last.round(2), atol=0.01)
