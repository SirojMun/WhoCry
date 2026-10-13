"""Pelatihan ulang mingguan di sistem realtime: strategi R4 (replay + gerbang champion–challenger).

1. Champion = model di <live>/models/current.
2. Challenger = champion yang di-fine-tune pada data ±10 minggu terakhir (Open-Meteo) + sampel replay
   historis (<live>/replay/pool.parquet), tanpa minggu terakhir.
3. Keduanya diuji pada minggu terakhir; challenger dipakai hanya jika >= margin lebih baik.
   Model lama diarsipkan ke models/archive/<tanggal>/ dan setiap keputusan dicatat di models/history.csv.
Scaler champion dipakai apa adanya (tidak di-fit ulang) agar challenger konsisten dengan champion.
"""

import copy
import shutil
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch

from whocry import settings
from whocry.data.openmeteo import fetch_archive, make_session
from whocry.features.build import WIND_IDX, WindowData, sample_index, time_features, to_array
from whocry.features.spatial import idw_weights, static_features, upwind_geometry
from whocry.realtime.predict import FETCH_VARS, load_artifact
from whocry.retrain import HOURS_PER_WEEK, RetrainConfig, _norm_mae, _train_epochs

RECENT_DAYS = 70          # 8 minggu latih + 1 minggu gerbang + cadangan jendela
MIN_DAYS_BETWEEN = 6      # workflow berjalan harian; pelatihan ulang hanya bila sudah ±seminggu
MODEL_FILES = ("config.json", "model.pt", "scaler.npz")


def fetch_recent(locations, days=RECENT_DAYS):
    """Data per jam `days` hari terakhir dari arsip Open-Meteo (model & sel grid sama dengan data latih)."""
    session = make_session()
    end = date.today()
    start = (end - timedelta(days=days)).isoformat()
    frames = []
    for loc in locations.itertuples():
        df = fetch_archive(session, loc.lat, loc.lon, start, end.isoformat(), FETCH_VARS, settings.OPENMETEO_MODEL)
        df.insert(0, "adm2", loc.adm2)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_data(hourly, locations, art):
    X, times = to_array(hourly, locations)
    data = WindowData(
        X=art["x_scaler"].transform(X), T_feat=time_features(times),
        static=art["s_scaler"].transform(static_features(locations)),
        W=idw_weights(locations["lat"], locations["lon"]), variant=art["variant"],
        wind=X[..., WIND_IDX].copy(), geo=upwind_geometry(locations["lat"], locations["lon"]))
    return data, times


def split(data, times, recent_start):
    """Indeks sampel gerbang (minggu terakhir), latih terbaru (sebelum gerbang), dan replay (historis)."""
    idx = sample_index(data.X, (0, len(times) - 1), n_in=data.n_in, n_out=data.n_out)
    rec_lo = times.searchsorted(recent_start)
    recent_all = idx[idx[:, 0] >= rec_lo]
    if len(recent_all) == 0:
        return idx[:0], idx[:0], idx[idx[:, 0] < rec_lo]
    gate_start = recent_all[:, 0].max() - HOURS_PER_WEEK + 1
    gate = recent_all[recent_all[:, 0] >= gate_start]
    # Target latih berakhir sebelum target gerbang dimulai (tanpa tumpang tindih).
    recent = recent_all[recent_all[:, 0] < gate_start - data.n_out]
    return gate, recent, idx[idx[:, 0] < rec_lo]


def due(history_path, today=None):
    if not history_path.exists():
        return True
    last = pd.to_datetime(pd.read_csv(history_path)["tanggal"]).max().date()
    return ((today or date.today()) - last).days >= MIN_DAYS_BETWEEN


def run(live_dir, rcfg=None, seed=0, force=False, recent=None, log=print):
    """Satu siklus champion–challenger. `recent` dapat diisi DataFrame untuk pengujian (tanpa jaringan)."""
    rcfg = rcfg or RetrainConfig(strategy="R4")
    models = live_dir / "models"
    history = models / "history.csv"
    if not force and not due(history):
        log("pelatihan ulang belum jatuh tempo")
        return None
    locations = pd.read_csv(settings.LOCATIONS_CSV, dtype={"adm2": str, "adm4_bmkg": str})
    art = load_artifact(models / "current", locations)
    if art is None:
        log("belum ada model champion; dilewati")
        return None

    recent_df = fetch_recent(locations) if recent is None else recent
    pool = pd.read_parquet(live_dir / "replay" / "pool.parquet")
    pool["adm2"] = pool["adm2"].astype(str)
    hourly = pd.concat([pool, recent_df], ignore_index=True).drop_duplicates(["adm2", "time"], keep="last")
    data, times = build_data(hourly, locations, art)
    gate, recent_idx, replay = split(data, times, recent_df["time"].min())
    if len(gate) == 0 or len(recent_idx) == 0:
        log("data terbaru tidak cukup; dilewati")
        return None

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_rep = min(len(replay), int(len(recent_idx) * rcfg.replay_ratio))
    mix = np.concatenate([recent_idx, replay[rng.choice(len(replay), n_rep, replace=False)]])
    challenger = _train_epochs(copy.deepcopy(art["model"]), data, mix, rcfg.replay_epochs, rcfg.replay_lr,
                               rcfg.batch_size, rng)
    champ, chall = _norm_mae(art["model"], data, gate), _norm_mae(challenger, data, gate)
    accepted = bool(chall < (1 - rcfg.gate_margin) * champ)

    today = date.today().isoformat()
    if accepted:
        archive = models / "archive" / today.replace("-", "")
        archive.mkdir(parents=True, exist_ok=True)
        for f in MODEL_FILES:
            shutil.copy(models / "current" / f, archive / f)
        torch.save(challenger.cpu().state_dict(), models / "current" / "model.pt")

    row = dict(tanggal=today, gerbang_mulai=str(times[gate[:, 0].min()]), champion_mae=round(champ, 5),
               challenger_mae=round(chall, 5), perbaikan_pct=round(100 * (1 - chall / champ), 2),
               diterima=accepted, n_terbaru=len(recent_idx), n_replay=n_rep, n_gerbang=len(gate))
    pd.concat([pd.read_csv(history) if history.exists() else pd.DataFrame(), pd.DataFrame([row])]) \
        .to_csv(history, index=False)
    log(f"champion {champ:.4f} vs challenger {chall:.4f} ({row['perbaikan_pct']:+.2f}%) -> "
        f"{'DITERIMA, model diganti' if accepted else 'ditolak, champion tetap'}")
    return row
