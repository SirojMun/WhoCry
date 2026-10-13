"""Uji alur champion–challenger realtime dengan data sintetis (tanpa jaringan)."""

import json

import numpy as np
import pandas as pd
import torch

from test_features import make_hourly, make_locations
from whocry import settings
from whocry.features.build import prepare
from whocry.realtime import retrain_live
from whocry.train import TrainConfig, fit


def setup_live(tmp_path, monkeypatch, variant="S4"):
    loc = make_locations(4).assign(adm4_bmkg="x", nama="X")
    loc_csv = tmp_path / "locations.csv"
    loc.to_csv(loc_csv, index=False)
    monkeypatch.setattr(settings, "LOCATIONS_CSV", loc_csv)
    hourly = make_hourly(loc, hours=24 * 90)
    data, index, meta = prepare(hourly, loc, variant, train_end="2020-02-15 23:00", val_end="2020-03-01 23:00")
    model, _ = fit(TrainConfig(model="gru", hidden=8, max_epochs=1, batch_size=64), data, index, log=lambda *_: None)
    cur = tmp_path / "live" / "models" / "current"
    cur.mkdir(parents=True)
    torch.save(model.state_dict(), cur / "model.pt")
    np.savez(cur / "scaler.npz", x_mean=meta["x_scaler"].mean, x_std=meta["x_scaler"].std,
             s_mean=meta["s_scaler"].mean, s_std=meta["s_scaler"].std)
    (cur / "config.json").write_text(json.dumps(dict(model="gru", variant=variant, n_features=data.n_features,
                                                     hidden=8, layers=1, dropout=0.1)))
    split_t = hourly["time"].min() + pd.Timedelta(days=30)
    (tmp_path / "live" / "replay").mkdir()
    hourly[hourly["time"] < split_t].to_parquet(tmp_path / "live" / "replay" / "pool.parquet")
    return tmp_path / "live", hourly[hourly["time"] >= split_t]


def test_cycle_writes_history_and_respects_schedule(tmp_path, monkeypatch):
    live, recent = setup_live(tmp_path, monkeypatch)
    rc = retrain_live.RetrainConfig(strategy="R4", batch_size=64)
    row = retrain_live.run(live, rcfg=rc, force=True, recent=recent, log=lambda *_: None)
    assert row is not None and row["n_gerbang"] > 0 and row["n_terbaru"] > 0
    hist = pd.read_csv(live / "models" / "history.csv")
    assert len(hist) == 1
    archived = (live / "models" / "archive").exists()
    assert archived == bool(row["diterima"])            # diarsipkan hanya jika challenger dipakai
    # Belum seminggu -> tidak jalan lagi tanpa --force.
    assert retrain_live.run(live, rcfg=rc, recent=recent, log=lambda *_: None) is None


def test_gate_and_training_do_not_overlap(tmp_path, monkeypatch):
    live, recent = setup_live(tmp_path, monkeypatch)
    loc = pd.read_csv(settings.LOCATIONS_CSV, dtype={"adm2": str})
    art = retrain_live.load_artifact(live / "models" / "current", loc)
    pool = pd.read_parquet(live / "replay" / "pool.parquet")
    data, times = retrain_live.build_data(pd.concat([pool, recent]), loc, art)
    gate, rec, replay = retrain_live.split(data, times, recent["time"].min())
    assert rec[:, 0].max() + data.n_out < gate[:, 0].min() + 1    # target latih berakhir sebelum target gerbang
    assert replay[:, 0].max() < rec[:, 0].min()
