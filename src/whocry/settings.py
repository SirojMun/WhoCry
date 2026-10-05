"""Konstanta proyek yang dipakai bersama oleh skrip, pipeline, dan notebook."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
LOCATIONS_CSV = ROOT / "config" / "locations.csv"

OPENMETEO_MODEL = "ecmwf_ifs"
HISTORY_START = "2017-01-01"  # data ECMWF IFS di Open-Meteo mulai tanggal ini (UTC)

TARGETS = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "precipitation",
]
SUPPORT_VARS = [
    "dew_point_2m",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
]
EXTRA_VARS = ["weather_code"]
HOURLY_VARS = TARGETS + SUPPORT_VARS + EXTRA_VARS

# Pembagian data kronologis (inklusif, UTC)
TRAIN_END = "2023-12-31 23:00"
VAL_END = "2024-12-31 23:00"

INPUT_HOURS = 24
HORIZON_HOURS = 24
