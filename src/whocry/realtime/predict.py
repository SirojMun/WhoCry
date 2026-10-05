"""Pipeline realtime per jam: ambil data terbaru -> prediksi 24 jam -> simpan.

Model diambil dari folder artefak (config.json, model.pt, scaler.npz) hasil
scripts/run_experiment.py. Jika belum ada model, dipakai persistence.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from whocry import settings
from whocry.data.openmeteo import FORECAST_URL, _get, make_session
from whocry.features.build import (TARGET_NAMES, Scaler, WindowData, inverse_targets, time_features,
                                   to_array)
from whocry.features.spatial import idw_weights, static_features
from whocry.models.nets import build_model

FETCH_VARS = [v for v in settings.HOURLY_VARS if v != "weather_code"]


def fetch_latest(locations, past_days=2, session=None):
    """Data per jam untuk semua lokasi dalam satu request, dipotong pada jam sekarang (UTC)."""
    session = session or make_session()
    payload = _get(session, FORECAST_URL, dict(
        latitude=",".join(map(str, locations["lat"])), longitude=",".join(map(str, locations["lon"])),
        hourly=",".join(FETCH_VARS), models=settings.OPENMETEO_MODEL,
        past_days=past_days, forecast_days=1, timezone="GMT"))
    payload = payload if isinstance(payload, list) else [payload]
    now = pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None)
    frames = []
    for adm2, d in zip(locations["adm2"], payload):
        h = pd.DataFrame(d["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        h.insert(0, "adm2", adm2)
        frames.append(h[h["time"] <= now])
    return pd.concat(frames, ignore_index=True), now


def load_artifact(model_dir, locations):
    """Muat model terlatih; kembalikan None jika folder tidak lengkap."""
    model_dir = Path(model_dir) if model_dir else None
    if not model_dir or not (model_dir / "model.pt").exists():
        return None
    cfg = json.loads((model_dir / "config.json").read_text())
    sc = np.load(model_dir / "scaler.npz")
    hp = {k: cfg[k] for k in ("hidden", "layers", "dropout")} if cfg["model"] in ("lstm", "gru") else {}
    model = build_model(cfg["model"], settings.INPUT_HOURS, cfg["n_features"],
                        settings.HORIZON_HOURS, len(TARGET_NAMES), **hp)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    return dict(model=model.eval(), variant=cfg["variant"], name=f"{cfg['model']}_{cfg['variant']}",
                x_scaler=Scaler(sc["x_mean"], sc["x_std"]), s_scaler=Scaler(sc["s_mean"], sc["s_std"]))


def forecast(hourly, locations, artifact=None):
    """Prediksi 24 jam ke depan dari jam terakhir data, untuk setiap lokasi."""
    X, times = to_array(hourly, locations)
    X = X[-settings.INPUT_HOURS:]
    times = times[-settings.INPUT_HOURS:]
    if artifact is None:  # persistence: nilai terakhir diulang 24 jam
        last = inverse_targets(X[-1:, :, : len(TARGET_NAMES)], Scaler(np.zeros(1, "float32"), np.ones(1, "float32")))
        preds = np.repeat(last.transpose(1, 0, 2), settings.HORIZON_HOURS, axis=1)
        model_name = "persistence"
    else:
        data = WindowData(
            X=artifact["x_scaler"].transform(X), T_feat=time_features(times),
            static=artifact["s_scaler"].transform(static_features(locations)),
            W=idw_weights(locations["lat"], locations["lon"]), variant=artifact["variant"])
        idx = np.column_stack([np.full(len(locations), len(times) - 1), np.arange(len(locations))])
        x = data.batch_inputs(idx)
        with torch.no_grad():
            y = artifact["model"](torch.from_numpy(x)).numpy()
        preds = inverse_targets(y, artifact["x_scaler"])
        model_name = artifact["name"]

    issued = times[-1]
    rows = []
    for j, adm2 in enumerate(locations["adm2"]):
        for h in range(settings.HORIZON_HOURS):
            row = dict(issued_at=issued, adm2=adm2, horizon=h + 1,
                       valid_time=issued + pd.Timedelta(hours=h + 1), model=model_name)
            row.update({name: round(float(preds[j, h, k]), 2) for k, name in enumerate(TARGET_NAMES)})
            rows.append(row)
    return pd.DataFrame(rows)
