"""Dasbor pemantauan cuaca realtime Jawa Tengah (Streamlit).

Lokal:   streamlit run dashboard/app.py
Data dibaca dari branch data-live dan data-bmkg di GitHub. Untuk uji lokal,
set WHOCRY_LIVE_DIR / WHOCRY_BMKG_DIR ke folder lokal.
"""

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO_RAW = "https://raw.githubusercontent.com/SirojMun/akuKehujanan"
LIVE = os.environ.get("WHOCRY_LIVE_DIR", f"{REPO_RAW}/data-live")
BMKG = os.environ.get("WHOCRY_BMKG_DIR", f"{REPO_RAW}/data-bmkg")
LOCATIONS = Path(__file__).resolve().parents[1] / "config" / "locations.csv"
WIB = pd.Timedelta(hours=7)

VARS = {
    "temperature_2m": ("Suhu", "°C", "t"),
    "relative_humidity_2m": ("Kelembapan", "%", "hu"),
    "cloud_cover": ("Tutupan awan", "%", "tcc"),
    "precipitation": ("Curah hujan", "mm/jam", "tp"),
}
THRESHOLDS = {"precipitation": (">=", 10.0), "temperature_2m": (">=", 35.0),
              "relative_humidity_2m": ("<=", 40.0)}


@st.cache_data(ttl=600)
def load():
    loc = pd.read_csv(LOCATIONS, dtype={"adm2": str})
    pred = pd.read_csv(f"{LIVE}/latest_predictions.csv", dtype={"adm2": str},
                       parse_dates=["issued_at", "valid_time"])
    obs = pd.read_csv(f"{LIVE}/latest_observations.csv", dtype={"adm2": str}, parse_dates=["time"])
    try:
        bmkg = pd.read_csv(f"{BMKG}/bmkg/latest.csv.gz", dtype={"adm2": str},
                           parse_dates=["utc_datetime"])
    except Exception:  # dasbor tetap jalan walau data BMKG belum tersedia
        bmkg = pd.DataFrame()
    for df, col in ((pred, "valid_time"), (obs, "time")):
        df["waktu_wib"] = df[col] + WIB
    if not bmkg.empty:
        bmkg["waktu_wib"] = bmkg["utc_datetime"] + WIB
    return loc, pred, obs, bmkg


def color_for(values, lo, hi):
    """Warna biru -> merah sesuai nilai (untuk peta)."""
    frac = ((values - lo) / max(hi - lo, 1e-6)).clip(0, 1)
    return [f"#{int(40 + 200 * f):02x}{int(90 + 40 * (1 - abs(2 * f - 1))):02x}{int(220 - 180 * f):02x}"
            for f in frac]


def main():
    st.set_page_config(page_title="Cuaca Jawa Tengah", page_icon="🌦️", layout="wide")
    st.title("🌦️ Prediksi Cuaca Realtime — Jawa Tengah")
    try:
        loc, pred, obs, bmkg = load()
    except Exception as e:
        st.error(f"Data realtime belum tersedia: {e}")
        return

    names = loc.set_index("adm2")["nama"]
    issued = pred["issued_at"].max() + WIB
    model = pred["model"].iloc[0]
    st.caption(f"Diterbitkan {issued:%d %b %Y %H:%M} WIB · model: **{model}** · "
               f"35 kabupaten/kota · horizon 1–24 jam")

    var = st.sidebar.radio("Variabel", list(VARS), format_func=lambda v: f"{VARS[v][0]} ({VARS[v][1]})")
    label, unit, bmkg_col = VARS[var]
    adm2 = st.sidebar.selectbox("Kabupaten/kota", loc["adm2"], format_func=lambda a: names[a])
    horizon = st.sidebar.select_slider("Horizon peta (jam ke depan)", [1, 3, 6, 12, 24], value=3)

    # --- Peta dan tabel ringkas -------------------------------------------------
    now_obs = obs[obs["time"] == obs["time"].max()].set_index("adm2")[var]
    at_h = pred[pred["horizon"] == horizon].set_index("adm2")[var]
    table = loc[["adm2", "nama", "lat", "lon"]].assign(
        sekarang=loc["adm2"].map(now_obs), prediksi=loc["adm2"].map(at_h))
    lo, hi = table["prediksi"].min(), table["prediksi"].max()
    table["warna"] = color_for(table["prediksi"], lo, hi)

    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"{label} — prediksi +{horizon} jam")
        st.map(table, latitude="lat", longitude="lon", color="warna", size=4000)
        st.caption(f"Biru = {lo:.1f} {unit}, merah = {hi:.1f} {unit}")
    with right:
        st.subheader("Semua lokasi")
        st.dataframe(table[["nama", "sekarang", "prediksi"]].rename(columns={
            "nama": "Kabupaten/kota", "sekarang": f"Sekarang ({unit})", "prediksi": f"+{horizon} jam ({unit})"}),
            hide_index=True, height=420)

    # --- Deret waktu lokasi terpilih ---------------------------------------------
    st.subheader(f"{names[adm2]} — {label}")
    o = obs[obs["adm2"] == adm2][["waktu_wib", var]].assign(sumber="Data (Open-Meteo)")
    p = pred[pred["adm2"] == adm2][["waktu_wib", var]].assign(sumber=f"Prediksi ({model})")
    parts = [o, p]
    if not bmkg.empty:
        b = bmkg[bmkg["adm2"] == adm2][["waktu_wib", bmkg_col]].rename(columns={bmkg_col: var})
        parts.append(b.assign(sumber="Prakiraan BMKG"))
    series = pd.concat(parts, ignore_index=True)
    chart = alt.Chart(series).mark_line(point=True).encode(
        x=alt.X("waktu_wib:T", title="Waktu (WIB)"),
        y=alt.Y(f"{var}:Q", title=f"{label} ({unit})"),
        color=alt.Color("sumber:N", title=None, legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("sumber:N", legend=None),
        tooltip=["sumber", alt.Tooltip("waktu_wib:T", format="%d %b %H:%M"), alt.Tooltip(f"{var}:Q", format=".1f")],
    ).properties(height=320)
    rule = alt.Chart(pd.DataFrame({"t": [issued]})).mark_rule(color="gray").encode(x="t:T")
    st.altair_chart(chart + rule, width="stretch")

    # --- Peringatan ---------------------------------------------------------------
    st.subheader("⚠️ Peringatan 24 jam ke depan")
    warnings = []
    for v, (op, thr) in THRESHOLDS.items():
        hit = pred[pred[v] >= thr] if op == ">=" else pred[pred[v] <= thr]
        for a, g in hit.groupby("adm2"):
            r = g.loc[g[v].idxmax() if op == ">=" else g[v].idxmin()]
            warnings.append({"Kabupaten/kota": names[a], "Peringatan": VARS[v][0],
                             "Nilai": f"{r[v]:.1f} {VARS[v][1]}", "Waktu (WIB)": f"{r['waktu_wib']:%d/%m %H:%M}"})
    if warnings:
        st.dataframe(pd.DataFrame(warnings), hide_index=True)
    else:
        st.success("Tidak ada prediksi yang melewati ambang peringatan.")

    st.divider()
    st.caption("Data: Open-Meteo (ECMWF IFS, CC-BY 4.0) · Prakiraan pembanding: BMKG · "
               "Penelitian skripsi — github.com/SirojMun/akuKehujanan")


main()
