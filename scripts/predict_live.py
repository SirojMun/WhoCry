"""Satu siklus prediksi realtime (dijalankan tiap jam oleh .github/workflows/predict.yml).

    python scripts/predict_live.py --out <folder data-live> [--model-dir <folder model>]

Menulis ke <out>:
  latest_predictions.csv                 prediksi 24 jam terbaru
  latest_observations.csv                data 48 jam terakhir (untuk dasbor)
  predictions/YYYY/MM/DD/HH.csv.gz       arsip prediksi
  observations/YYYY/MM/DD/HH.csv.gz      arsip data jam terakhir (kebenaran dasar realtime)
  alerts_state.json                      riwayat peringatan
"""

import argparse
from pathlib import Path

import pandas as pd

from whocry import settings
from whocry.realtime.alerts import process
from whocry.realtime.predict import fetch_latest, forecast, load_artifact

OBS_COLS = ["adm2", "time"] + settings.TARGETS + ["wind_speed_10m", "wind_direction_10m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, default=None)
    args = ap.parse_args()

    locations = pd.read_csv(settings.LOCATIONS_CSV, dtype={"adm2": str, "adm4_bmkg": str})
    hourly, now = fetch_latest(locations)
    artifact = load_artifact(args.model_dir or args.out / "models" / "current", locations)
    pred = forecast(hourly, locations, artifact)

    out = args.out
    stamp = pd.Timestamp(pred["issued_at"].iloc[0])
    sub = stamp.strftime("%Y/%m/%d")
    for kind, df in (("predictions", pred),
                     ("observations", hourly.loc[hourly["time"] == stamp, OBS_COLS])):
        path = out / kind / sub / f"{stamp:%H}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    pred.to_csv(out / "latest_predictions.csv", index=False)
    recent = hourly[hourly["time"] > stamp - pd.Timedelta(hours=48)][OBS_COLS]
    recent.to_csv(out / "latest_observations.csv", index=False)

    new_events = process(pred, locations, out / "alerts_state.json")
    print(f"diterbitkan {stamp} UTC dengan {pred['model'].iloc[0]}; {len(pred)} baris; "
          f"peringatan baru: {len(new_events)}")


if __name__ == "__main__":
    main()
