"""Periksa satuan kecepatan angin (ws) dan periode akumulasi hujan (tp) BMKG.

Membandingkan snapshot prakiraan BMKG dengan prakiraan Open-Meteo (ECMWF IFS)
untuk lokasi dan jam yang sama.

    python scripts/check_bmkg_units.py <snapshot.csv.gz>
"""

import sys

import numpy as np
import pandas as pd

from whocry import settings
from whocry.data.openmeteo import FORECAST_URL, make_session


def main(path):
    bmkg = pd.read_csv(path, dtype={"adm2": str}, parse_dates=["utc_datetime"])
    loc = pd.read_csv(settings.LOCATIONS_CSV, dtype={"adm2": str})
    r = make_session().get(FORECAST_URL, params=dict(
        latitude=",".join(map(str, loc["lat"])), longitude=",".join(map(str, loc["lon"])),
        hourly="wind_speed_10m,precipitation,temperature_2m,relative_humidity_2m,cloud_cover",
        models=settings.OPENMETEO_MODEL, past_days=1, forecast_days=4, timezone="GMT"), timeout=120)
    r.raise_for_status()
    frames = []
    for adm2, d in zip(loc["adm2"], r.json()):
        h = pd.DataFrame(d["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        h = h.set_index("time")
        h["precip_3h"] = h["precipitation"].rolling(3).sum()   # akumulasi 3 jam ke belakang
        h["precip_6h"] = h["precipitation"].rolling(6).sum()
        h["adm2"] = adm2
        frames.append(h.reset_index())
    om = pd.concat(frames)
    m = bmkg.merge(om, left_on=["adm2", "utc_datetime"], right_on=["adm2", "time"], how="inner")
    print(f"pasangan cocok: {len(m)}")

    ratio = (m["ws"] / m["wind_speed_10m"]).median()
    print(f"\nws BMKG / angin Open-Meteo (km/jam), median rasio = {ratio:.2f}")
    print("  ≈1,00 → km/jam   ≈0,28 → m/s   ≈0,54 → knot")
    for col in ["precipitation", "precip_3h", "precip_6h"]:
        c = np.corrcoef(m["tp"], m[col])[0, 1]
        print(f"korelasi tp vs Open-Meteo {col:14s}: r = {c:.2f}  (rerata {m[col].mean():.2f} vs tp {m['tp'].mean():.2f})")
    for b, o in [("t", "temperature_2m"), ("hu", "relative_humidity_2m"), ("tcc", "cloud_cover")]:
        print(f"{b:4s} vs {o:22s}: r = {np.corrcoef(m[b], m[o])[0, 1]:.2f}, "
              f"selisih rerata = {(m[b] - m[o]).mean():+.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
