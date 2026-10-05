"""Unduh data historis Open-Meteo (ECMWF IFS) untuk 35 lokasi, satu file per lokasi per tahun.

Dapat dilanjutkan: tahun yang sudah lengkap dilewati. Laju dibatasi agar tidak
melewati kuota per jam. Setelah selesai, semua file digabung menjadi
data/processed/hourly.parquet.

    python scripts/download_history.py            # unduh + gabung
    python scripts/download_history.py --merge    # gabung saja
"""

import argparse
import time
from collections import deque
from datetime import date, timedelta

import pandas as pd

from whocry import settings
from whocry.data.openmeteo import fetch_archive, make_session, request_weight

RAW_DIR = settings.DATA_DIR / "raw" / "openmeteo"
OUT = settings.DATA_DIR / "processed" / "hourly.parquet"
BUDGETS = ((60, 500), (3600, 4000))  # (detik, bobot): di bawah batas 600/menit dan 5.000/jam


class Throttle:
    """Menjaga total bobot request di setiap jendela waktu tetap di bawah anggarannya."""

    def __init__(self, budgets):
        self.budgets = budgets
        self.log = deque()

    def wait(self, weight):
        horizon = max(sec for sec, _ in self.budgets)
        while True:
            now = time.time()
            while self.log and now - self.log[0][0] > horizon:
                self.log.popleft()
            pause = 0.0
            for sec, budget in self.budgets:
                recent = [(t, w) for t, w in self.log if now - t <= sec]
                used = sum(w for _, w in recent)
                if used + weight > budget:
                    pause = max(pause, sec - (now - recent[0][0]) + 1)
            if pause == 0:
                self.log.append((now, weight))
                return
            if pause > 120:
                print(f"  anggaran kuota terpakai; tunggu {pause / 60:.1f} menit", flush=True)
            time.sleep(pause)


def year_ranges(end_date):
    start = date.fromisoformat(settings.HISTORY_START)
    for year in range(start.year, end_date.year + 1):
        s = max(start, date(year, 1, 1))
        e = min(end_date, date(year, 12, 31))
        yield year, s.isoformat(), e.isoformat()


def is_complete(path, end):
    if not path.exists():
        return False
    last = pd.read_parquet(path, columns=["time"])["time"].max()
    return last >= pd.Timestamp(end) + pd.Timedelta(hours=23)


def download(locations):
    end_date = date.today() - timedelta(days=1)
    session = make_session()
    throttle = Throttle(BUDGETS)
    tasks = [(loc, *yr) for loc in locations.itertuples() for yr in year_ranges(end_date)]
    total_w = sum(request_weight(s, e, len(settings.HOURLY_VARS)) for _, _, s, e in tasks)
    print(f"{len(tasks)} file, total bobot ≈ {total_w:.0f} panggilan", flush=True)

    for i, (loc, year, start, end) in enumerate(tasks, 1):
        path = RAW_DIR / loc.adm2 / f"{year}.parquet"
        if is_complete(path, end):
            continue
        w = request_weight(start, end, len(settings.HOURLY_VARS))
        throttle.wait(w)
        df = fetch_archive(session, loc.lat, loc.lon, start, end, settings.HOURLY_VARS,
                           settings.OPENMETEO_MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        n_null = int(df[settings.TARGETS].isna().sum().sum())
        print(f"[{i}/{len(tasks)}] {loc.adm2} {year}: {len(df)} baris, null target={n_null}", flush=True)


def merge(locations):
    frames = []
    for loc in locations.itertuples():
        files = sorted((RAW_DIR / loc.adm2).glob("*.parquet"))
        if not files:
            print(f"!! {loc.adm2} belum ada data")
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df.insert(0, "adm2", loc.adm2)
        frames.append(df)
    data = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(["adm2", "time"])
            .sort_values(["adm2", "time"])
            .reset_index(drop=True))
    data["adm2"] = data["adm2"].astype("category")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUT, index=False)
    print(f"tersimpan {OUT}: {len(data):,} baris, {data['adm2'].nunique()} lokasi, "
          f"{data['time'].min()} – {data['time'].max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="hanya gabungkan file yang sudah ada")
    args = ap.parse_args()
    locations = pd.read_csv(settings.LOCATIONS_CSV, dtype={"adm2": str, "adm4_bmkg": str})
    if not args.merge:
        download(locations)
    merge(locations)


if __name__ == "__main__":
    main()
