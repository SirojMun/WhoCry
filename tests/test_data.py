import pandas as pd
import pytest

from whocry.data.bmkg import parse_forecast
from whocry.data.openmeteo import _to_frame, request_weight

BMKG_SAMPLE = {
    "lokasi": {"adm4": "33.74.01.1001"},
    "data": [{
        "cuaca": [
            [{"utc_datetime": "2026-10-05 16:00:00", "analysis_date": "2026-10-05T12:00:00",
              "t": 28, "hu": 78, "tcc": 80, "tp": 0.1, "ws": 2, "wd_deg": 255,
              "weather": 2, "weather_desc": "Cerah Berawan", "vs": None}],
            [{"utc_datetime": "2026-10-05 19:00:00", "analysis_date": "2026-10-05T12:00:00",
              "t": 27, "hu": 82, "tcc": 90, "tp": 0.0, "ws": 3, "wd_deg": 240,
              "weather": 3, "weather_desc": "Berawan", "vs": None},
             {"utc_datetime": "2026-10-05 22:00:00", "analysis_date": "2026-10-05T12:00:00",
              "t": 26, "hu": 85, "tcc": 95, "tp": 1.2, "ws": 1, "wd_deg": 200,
              "weather": 61, "weather_desc": "Hujan Sedang", "vs": None}],
        ]
    }],
}


def test_parse_forecast_flattens_days_and_slots():
    df = parse_forecast(BMKG_SAMPLE)
    assert len(df) == 3
    assert df["utc_datetime"].is_monotonic_increasing
    assert df["t"].tolist() == [28, 27, 26]
    assert pd.api.types.is_datetime64_any_dtype(df["analysis_date"])


def test_parse_forecast_empty_payload():
    assert parse_forecast({}).empty


@pytest.mark.parametrize("start,end,n_vars,expected", [
    ("2024-01-01", "2024-01-14", 10, 1.0),     # 2 minggu, 10 variabel = 1 panggilan
    ("2024-01-01", "2024-01-28", 10, 2.0),
    ("2024-01-01", "2024-01-14", 15, 1.5),
    ("2024-01-01", "2024-01-01", 3, 1.0),      # minimal 1
])
def test_request_weight(start, end, n_vars, expected):
    assert request_weight(start, end, n_vars) == pytest.approx(expected)


def test_openmeteo_frame_types():
    payload = {"hourly": {"time": ["2017-01-01T00:00", "2017-01-01T01:00"],
                          "temperature_2m": [25.1, None]}}
    df = _to_frame(payload, ["temperature_2m"])
    assert df["temperature_2m"].dtype == "float32"
    assert df["temperature_2m"].isna().sum() == 1
    assert df["time"].iloc[1] == pd.Timestamp("2017-01-01 01:00")
