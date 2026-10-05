"""Kolektor prakiraan cuaca publik BMKG (api.bmkg.go.id), dipakai sebagai pembanding.

Satu request per kelurahan (kode adm4); batas 60 request/menit per IP.
Data wajib mencantumkan BMKG sebagai sumber.
"""

import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://api.bmkg.go.id/publik/prakiraan-cuaca"
FIELDS = ["t", "hu", "tcc", "tp", "ws", "wd_deg", "weather", "weather_desc", "vs"]


def make_session():
    s = requests.Session()
    s.headers["User-Agent"] = "WhoCry-skripsi/0.1 (github.com/SirojMun/akuKehujanan)"
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=3, status_forcelist=(429, 500, 502, 503, 504))))
    return s


def parse_forecast(payload):
    """Ratakan respons BMKG (daftar hari -> daftar slot 3 jam) menjadi tabel."""
    rows = []
    for block in payload.get("data", []):
        for day in block.get("cuaca", []):
            for slot in day:
                row = {k: slot.get(k) for k in FIELDS}
                row["analysis_date"] = slot.get("analysis_date")
                row["utc_datetime"] = slot.get("utc_datetime")
                rows.append(row)
    df = pd.DataFrame(rows, columns=["analysis_date", "utc_datetime", *FIELDS])
    for col in ("analysis_date", "utc_datetime"):
        df[col] = pd.to_datetime(df[col])
    return df


def collect(locations, session=None, pause_s=1.1):
    """Ambil prakiraan untuk semua lokasi. Lokasi yang gagal dilewati dan dilaporkan."""
    session = session or make_session()
    frames, failed = [], []
    for loc in locations.itertuples():
        try:
            r = session.get(URL, params={"adm4": loc.adm4_bmkg}, timeout=30, allow_redirects=False)
            r.raise_for_status()
            df = parse_forecast(r.json())
            df.insert(0, "adm4", loc.adm4_bmkg)
            df.insert(0, "adm2", loc.adm2)
            frames.append(df)
        except (requests.RequestException, ValueError) as e:
            failed.append((loc.adm2, str(e)[:120]))
        time.sleep(pause_s)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return data, failed
