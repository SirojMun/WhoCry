"""Dasbor pemantauan cuaca realtime Jawa Tengah & DIY (Streamlit).

Lokal:   streamlit run dashboard/app.py
Prediksi & data terkini dibaca dari branch data-live dan data-bmkg di GitHub; riwayat
sampai 1 tahun diambil langsung dari Open-Meteo per kabupaten yang dipilih.
Untuk uji lokal, set WHOCRY_LIVE_DIR / WHOCRY_BMKG_DIR ke folder lokal.
"""

import json
import os
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

REPO_RAW = "https://raw.githubusercontent.com/SirojMun/akuKehujanan"
LIVE = os.environ.get("WHOCRY_LIVE_DIR", f"{REPO_RAW}/data-live")
BMKG = os.environ.get("WHOCRY_BMKG_DIR", f"{REPO_RAW}/data-bmkg")
CONFIG = Path(__file__).resolve().parents[1] / "config"
WIB = pd.Timedelta(hours=7)

# kolom: (label, satuan, kolom BMKG)
VARS = {
    "temperature_2m": ("Suhu", "°C", "t"),
    "relative_humidity_2m": ("Kelembapan", "%", "hu"),
    "cloud_cover": ("Tutupan awan", "%", "tcc"),
    "precipitation": ("Curah hujan", "mm/jam", "tp"),
}
MAP_HTML = (Path(__file__).parent / "wind_map.html").read_text()
ICONS = [("☀️", "cerah"), ("🌤️", "cerah berawan"), ("⛅", "berawan"), ("☁️", "mendung"),
         ("🌧️", "hujan"), ("⛈️", "hujan lebat")]


def cuaca(cc, p, jam_wib):
    """Kondisi cuaca berbasis aturan dari tutupan awan (%) dan curah hujan (mm/jam)."""
    if p >= 10:
        return ICONS[5]
    if p >= 0.1:
        return ICONS[4]
    icon = ICONS[0] if cc < 20 else ICONS[1] if cc < 60 else ICONS[2] if cc < 90 else ICONS[3]
    malam = jam_wib >= 18 or jam_wib < 6
    return ("🌙", "cerah") if malam and icon == ICONS[0] else icon
THRESHOLDS = {"precipitation": (">=", 10.0), "temperature_2m": (">=", 35.0),
              "relative_humidity_2m": ("<=", 40.0)}
RANGES = {"24 jam": 1, "3 hari": 3, "7 hari": 7, "30 hari": 30, "90 hari": 90, "1 tahun": 365}
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
         "September", "Oktober", "November", "Desember"]
BULAN_JS = "['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']"


def tgl(ts, jam=True):
    """Tanggal berbahasa Indonesia, mis. 'Kamis, 8 Oktober 2026 20.00'."""
    s = f"{HARI[ts.weekday()]}, {ts.day} {BULAN[ts.month - 1]} {ts.year}"
    return f"{s} {ts:%H.%M}" if jam else s


def x_axis(days):
    """Sumbu waktu berlabel bahasa Indonesia (tanpa AM/PM)."""
    label = f"date(datum.value) + ' ' + {BULAN_JS}[month(datum.value)]"
    if days <= 3:
        label += " + ' ' + timeFormat(datum.value, '%H.%M')"
    return alt.X("waktu:T", title=None, axis=alt.Axis(labelExpr=label, labelAngle=0, tickCount=8))


def short(nama):
    return nama.replace("Kabupaten ", "Kab. ")


@st.cache_data(ttl=600)
def load_live():
    loc = pd.read_csv(CONFIG / "locations.csv", dtype={"adm2": str})
    pred = pd.read_csv(f"{LIVE}/latest_predictions.csv", dtype={"adm2": str},
                       parse_dates=["issued_at", "valid_time"])
    obs = pd.read_csv(f"{LIVE}/latest_observations.csv", dtype={"adm2": str}, parse_dates=["time"])
    try:
        bmkg = pd.read_csv(f"{BMKG}/bmkg/latest.csv.gz", dtype={"adm2": str}, parse_dates=["utc_datetime"])
        bmkg["waktu"] = bmkg["utc_datetime"] + WIB
    except Exception:  # dasbor tetap jalan walau data BMKG belum tersedia
        bmkg = pd.DataFrame()
    try:
        wind = requests.get(f"{LIVE}/wind.json", timeout=30).json() if LIVE.startswith("http") \
            else json.loads(Path(LIVE, "wind.json").read_text())
    except Exception:  # peta tetap tampil tanpa animasi angin
        wind = None
    pred["waktu"] = pred["valid_time"] + WIB
    obs["waktu"] = obs["time"] + WIB
    return loc, pred, obs, bmkg, wind


@st.cache_data
def load_geo():
    return json.loads((CONFIG / "wilayah.geojson").read_text())["features"]


@st.cache_data(ttl=3 * 3600, show_spinner="Mengambil riwayat dari Open-Meteo…")
def load_history(lat, lon):
    """Riwayat per jam 1 tahun terakhir (ECMWF IFS, sumber yang sama dengan data latih)."""
    end = date.today()
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params=dict(
        latitude=lat, longitude=lon, start_date=(end - timedelta(days=366)).isoformat(),
        end_date=end.isoformat(), hourly=",".join(VARS), models="ecmwf_ifs", timezone="GMT"))
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["waktu"] = pd.to_datetime(h.pop("time")) + WIB
    return h.dropna()


def wind_map(features, table, title, wind, frame_time, animate, malam):
    """Satu peta untuk empat variabel + partikel angin (komponen HTML/d3)."""
    w = None
    if wind:  # frame angin terdekat dengan waktu peta
        times = pd.to_datetime(wind["times"])
        i = int(abs(times - frame_time).argmin())
        w = dict(lats=wind["lats"], lons=wind["lons"], u=wind["u"][i], v=wind["v"][i])
    values = {r.adm2: dict(t=r.t, rh=r.rh, cc=r.cc, p=r.p, icon=r.icon, cuaca=r.cuaca)
              for r in table.dropna(subset=["t"]).itertuples()}
    icons = [("🌙", "cerah")] + ICONS[1:] if malam else ICONS
    data = dict(features=features, title=title, values=values, icons=icons, wind=w, animate=animate, height=560)
    st.iframe(MAP_HTML.replace("/*DATA*/null", json.dumps(data)), height=570)  # data dari pipeline sendiri


def main():
    st.set_page_config(page_title="Cuaca Jawa Tengah & DIY", page_icon="🌦️", layout="wide")
    st.title("🌦️ Prediksi Cuaca Realtime — Jawa Tengah & DIY")
    try:
        loc, pred, obs, bmkg, wind = load_live()
    except Exception as e:
        st.error(f"Data realtime belum tersedia: {e}")
        return
    names = loc.set_index("adm2")["nama"]
    issued = pred["issued_at"].max() + WIB
    model = pred["model"].iloc[0]
    st.caption(f"Diterbitkan {tgl(issued)} WIB · model: **{model}** · {len(loc)} kabupaten/kota")

    # --- Filter --------------------------------------------------------------------
    sb = st.sidebar
    sb.header("Peta")
    map_h = sb.select_slider("Waktu peta", [0, 1, 3, 6, 12, 24], value=0,
                             format_func=lambda h: "Sekarang" if h == 0 else f"+{h} jam")
    sb.header("Grafik")
    chosen = sb.multiselect("Kabupaten/kota", loc["adm2"], default=["33.74"],
                            format_func=lambda a: names[a], max_selections=6)
    chart_vars = sb.multiselect("Variabel", list(VARS), default=["temperature_2m", "relative_humidity_2m"],
                                format_func=lambda v: VARS[v][0])
    rng = sb.selectbox("Rentang waktu", list(RANGES), index=2)
    show_pred = sb.checkbox("Tampilkan prediksi 24 jam", value=True)
    show_bmkg = sb.checkbox("Tampilkan prakiraan BMKG", value=False)
    sb.header("Tampilan peta")
    animate = sb.checkbox("Animasi angin", value=True)

    # --- Peta (suhu = warna, awan + hujan = ikon, kelembapan = angka, angin = partikel) ------------
    cols = {"temperature_2m": "t", "relative_humidity_2m": "rh", "cloud_cover": "cc", "precipitation": "p"}
    if map_h == 0:
        cur = obs[obs["time"] == obs["time"].max()].set_index("adm2")[list(cols)]
        frame_time = obs["time"].max()
    else:
        cur = pred[pred["horizon"] == map_h].set_index("adm2")[list(cols)]
        frame_time = pred["issued_at"].max() + pd.Timedelta(hours=map_h)
    when_wib = frame_time + WIB
    table = loc[["adm2", "nama"]].join(cur.rename(columns=cols), on="adm2")
    table[["icon", "cuaca"]] = [cuaca(cc, p, when_wib.hour) if pd.notna(cc) else ("", "")
                                for cc, p in zip(table["cc"], table["p"])]
    title = ("Sekarang — " if map_h == 0 else f"Prediksi +{map_h} jam — ") + f"{tgl(when_wib)} WIB"

    left, right = st.columns([3, 2])
    with left:
        wind_map(load_geo(), table, title, wind, frame_time, animate, malam=when_wib.hour >= 18 or when_wib.hour < 6)
    with right:
        st.dataframe(
            table.assign(cuaca=table["icon"] + " " + table["cuaca"]).sort_values("t", ascending=False)[
                ["nama", "cuaca", "t", "rh", "cc", "p"]].round(1),
            column_config={"nama": "Kabupaten/kota", "cuaca": "Cuaca", "t": "Suhu (°C)", "rh": "Kelembapan (%)",
                           "cc": "Awan (%)", "p": "Hujan (mm/jam)"},
            hide_index=True, height=570, width="stretch")

    # --- Grafik garis -------------------------------------------------------------------
    if not chosen or not chart_vars:
        st.info("Pilih minimal satu kabupaten/kota dan satu variabel di panel kiri.")
    else:
        days = RANGES[rng]
        start = obs["waktu"].max() - pd.Timedelta(days=days)
        parts = []
        for a in chosen:
            row = loc.set_index("adm2").loc[a]
            try:
                hist = load_history(row["lat"], row["lon"])
            except Exception as e:
                st.warning(f"Riwayat {names[a]} gagal diambil ({e}); hanya 48 jam terakhir yang ditampilkan.")
                hist = pd.DataFrame(columns=["waktu", *VARS])
            recent = obs[obs["adm2"] == a][["waktu", *VARS]]
            data = pd.concat([hist, recent]).drop_duplicates("waktu", keep="last")
            parts.append(data[data["waktu"] >= start].assign(wilayah=short(names[a]), jenis="Data"))
            if show_pred:
                parts.append(pred[pred["adm2"] == a][["waktu", *VARS]].assign(
                    wilayah=short(names[a]), jenis="Prediksi"))
            if show_bmkg and not bmkg.empty:
                b = bmkg[bmkg["adm2"] == a].rename(columns={v[2]: k for k, v in VARS.items()})
                parts.append(b[["waktu", *VARS]].assign(wilayah=short(names[a]), jenis="BMKG"))
        series = pd.concat(parts, ignore_index=True)
        daily = days >= 30
        if daily:  # ponytail: rata-rata harian agar grafik setahun tetap terbaca; ubah ambang bila perlu
            series = (series.assign(waktu=series["waktu"].dt.floor("D"))
                      .groupby(["wilayah", "jenis", "waktu"], as_index=False)[list(VARS)].mean())
        series["waktu_teks"] = series["waktu"].map(lambda t: tgl(t, jam=not daily))

        st.subheader(f"Grafik {rng} terakhir" + (" (rata-rata harian)" if daily else ""))
        for v in chart_vars:
            vl, vu = VARS[v][:2]
            line = alt.Chart(series).mark_line(strokeWidth=1.8, interpolate="monotone").encode(
                x=x_axis(days),
                y=alt.Y(f"{v}:Q", title=f"{vl} ({vu})", scale=alt.Scale(zero=v == "precipitation")),
                color=alt.Color("wilayah:N", title=None, legend=alt.Legend(orient="top")),
                strokeDash=alt.StrokeDash("jenis:N", title=None,
                                          scale=alt.Scale(domain=["Data", "Prediksi", "BMKG"],
                                                          range=[[1, 0], [6, 3], [2, 2]])),
                tooltip=[alt.Tooltip("wilayah:N", title="Wilayah"), alt.Tooltip("jenis:N", title="Jenis"),
                         alt.Tooltip("waktu_teks:N", title="Waktu (WIB)"),
                         alt.Tooltip(f"{v}:Q", title=f"{vl} ({vu})", format=".1f")],
            )
            now = alt.Chart(pd.DataFrame({"waktu": [issued]})).mark_rule(color="gray", strokeDash=[3, 3]) \
                .encode(x="waktu:T")
            st.altair_chart((line + now).properties(height=280, title=vl).interactive(bind_y=False),
                            width="stretch")
        st.caption("Garis utuh = data, putus-putus = prediksi model, titik-titik = prakiraan BMKG. "
                   "Garis vertikal = waktu terbit prediksi. Gulir/seret grafik untuk memperbesar.")

    # --- Peringatan ------------------------------------------------------------------------
    st.subheader("⚠️ Peringatan 24 jam ke depan")
    warnings = []
    for v, (op, thr) in THRESHOLDS.items():
        hit = pred[pred[v] >= thr] if op == ">=" else pred[pred[v] <= thr]
        for a, g in hit.groupby("adm2"):
            r = g.loc[g[v].idxmax() if op == ">=" else g[v].idxmin()]
            warnings.append({"Kabupaten/kota": names[a], "Peringatan": VARS[v][0],
                             "Nilai": f"{r[v]:.1f} {VARS[v][1]}", "Waktu (WIB)": tgl(r["waktu"])})
    if warnings:
        st.dataframe(pd.DataFrame(warnings), hide_index=True)
    else:
        st.success("Tidak ada prediksi yang melewati ambang peringatan.")

    st.divider()
    st.caption("Data: Open-Meteo (ECMWF IFS, CC-BY 4.0; angin = prakiraan ECMWF IFS) · Batas wilayah: geoBoundaries/BPS (CC BY 3.0 IGO) · "
               "Prakiraan pembanding: BMKG · Penelitian skripsi — github.com/SirojMun/akuKehujanan")


main()
