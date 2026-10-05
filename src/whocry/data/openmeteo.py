"""Pengambilan data per jam dari Open-Meteo (model ECMWF IFS).

Bobot kuota Open-Meteo per request = (hari / 14) x (variabel / 10).
Batas gratis: 600/menit, 5.000/jam, 10.000/hari.
"""

import time
from datetime import date

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def make_session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, connect=5, read=5, backoff_factor=2, status_forcelist=(500, 502, 503, 504))))
    return s


def request_weight(start, end, n_vars):
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    return max(1.0, days / 14) * max(1.0, n_vars / 10)


def _to_frame(payload, variables):
    hourly = payload["hourly"]
    df = pd.DataFrame({v: hourly[v] for v in variables})
    df.insert(0, "time", pd.to_datetime(hourly["time"]))
    return df.astype({v: "float32" for v in variables})


def _get(session, url, params, max_wait_s=3 * 3600):
    """GET dengan penanganan batas kuota: tunggu lalu coba lagi saat HTTP 429."""
    waited, failures = 0, 0
    while True:
        try:
            r = session.get(url, params=params, timeout=120)
        except (requests.ConnectionError, requests.Timeout) as e:
            # Putus di tengah transfer tidak ditangani oleh Retry bawaan urllib3.
            failures += 1
            if failures > 8:
                raise
            print(f"  koneksi gagal ({type(e).__name__}); coba lagi dalam {30 * failures} detik", flush=True)
            time.sleep(30 * failures)
            continue
        if r.status_code != 429:
            r.raise_for_status()
            return r.json()
        if waited >= max_wait_s:
            r.raise_for_status()
        reason = r.json().get("reason", "") if r.headers.get("content-type", "").startswith("application/json") else ""
        pause = 3600 if "Daily" in reason else 300 if "Hourly" in reason else 60
        print(f"  kuota habis ({reason or 429}); tunggu {pause // 60} menit", flush=True)
        time.sleep(pause)
        waited += pause


def fetch_archive(session, lat, lon, start, end, variables, model):
    params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                  hourly=",".join(variables), models=model, timezone="GMT")
    return _to_frame(_get(session, ARCHIVE_URL, params), variables)


def fetch_recent(session, lat, lon, variables, model, past_days=2, forecast_days=1):
    """Data terkini untuk pipeline realtime (sel grid sama dengan data historis)."""
    params = dict(latitude=lat, longitude=lon, hourly=",".join(variables), models=model,
                  past_days=past_days, forecast_days=forecast_days, timezone="GMT")
    return _to_frame(_get(session, FORECAST_URL, params), variables)
