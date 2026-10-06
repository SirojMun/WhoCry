"""Eksplorasi data (F2): ringkasan statistik dan gambar untuk Bab III/IV.

    python scripts/eda.py

Keluaran: reports/eda_summary.md dan reports/figures/eda_*.png
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.titlesize"] = 9
import numpy as np
import pandas as pd

from whocry import settings
from whocry.features.build import TARGET_NAMES
from whocry.features.spatial import coast_distances, haversine_km

REPORTS = settings.ROOT / "reports"
FIG = REPORTS / "figures"
LABELS = {"temperature_2m": "Suhu (°C)", "relative_humidity_2m": "Kelembapan (%)",
          "cloud_cover": "Tutupan awan (%)", "precipitation": "Curah hujan (mm/jam)"}
WIB = pd.Timedelta(hours=7)


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"eda_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def completeness(df, locations):
    expected = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    rows = []
    for adm2, g in df.groupby("adm2", observed=True):
        rows.append(dict(adm2=adm2, jam=len(g), jam_hilang=len(expected) - g["time"].nunique(),
                         nilai_kosong=int(g[settings.HOURLY_VARS].isna().sum().sum())))
    out = pd.DataFrame(rows).merge(locations[["adm2", "nama"]], on="adm2")
    return out, expected


def target_stats(df):
    s = df[TARGET_NAMES].describe(percentiles=[0.01, 0.5, 0.99]).T
    p = df["precipitation"]
    rain = {thr: float((p >= thr).mean()) for thr in (0.1, 1.0, 10.0)}
    rain["nol"] = float((p == 0).mean())
    return s, rain


def plot_distributions(df):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
    for ax, name in zip(axes, TARGET_NAMES):
        v = df[name].to_numpy()
        if name == "precipitation":
            ax.hist(v[v > 0], bins=np.logspace(-1, 2, 40), color="C0")
            ax.set_xscale("log")
            ax.set_title(f"{LABELS[name]}\n(hanya jam > 0; {np.mean(v == 0):.0%} jam = 0)")
        else:
            ax.hist(v, bins=50, color="C0")
            ax.set_title(LABELS[name])
        ax.grid(alpha=.3)
    savefig(fig, "distribusi")


def plot_cycles(df):
    local = df["time"] + WIB
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for k, name in enumerate(TARGET_NAMES):
        by_hour = df.groupby(local.dt.hour)[name].mean()
        by_month = df.groupby(local.dt.month)[name].mean()
        axes[0, k].plot(by_hour.index, by_hour.values, marker="o", ms=3)
        axes[0, k].set_title(f"{LABELS[name]} — siklus harian (WIB)")
        axes[1, k].bar(by_month.index, by_month.values)
        axes[1, k].set_title(f"{LABELS[name]} — rata-rata bulanan")
        for ax in axes[:, k]:
            ax.grid(alpha=.3)
    savefig(fig, "siklus")


def autocorrelation(df, max_lag=72):
    out = {}
    for name in TARGET_NAMES:
        acfs = []
        for _, g in df.groupby("adm2", observed=True):
            x = g[name].to_numpy(dtype="float64")
            x = x - x.mean()
            denom = (x * x).sum()
            acfs.append([1.0] + [(x[:-k] * x[k:]).sum() / denom for k in range(1, max_lag + 1)])
        out[name] = np.mean(acfs, axis=0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for name, acf in out.items():
        ax.plot(acf, label=LABELS[name])
    ax.axvline(24, color="k", ls="--", lw=.8)
    ax.set_xlabel("lag (jam)"); ax.set_ylabel("autokorelasi"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    savefig(fig, "autokorelasi")
    return {n: dict(lag1=a[1], lag3=a[3], lag6=a[6], lag12=a[12], lag24=a[24]) for n, a in out.items()}


def spatial_correlation(df, locations):
    """Korelasi anomali (dikurangi rata-rata per jam-hari & bulan) antarlokasi terhadap jarak."""
    loc = locations.set_index("adm2")
    order = loc.index.tolist()
    d = haversine_km(loc["lat"].to_numpy()[:, None], loc["lon"].to_numpy()[:, None],
                     loc["lat"].to_numpy()[None, :], loc["lon"].to_numpy()[None, :])
    iu = np.triu_indices(len(order), 1)
    local = df["time"] + WIB
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2), sharey=True)
    summary = {}
    for ax, name in zip(axes, TARGET_NAMES):
        clim = df.groupby([df["adm2"], local.dt.month, local.dt.hour], observed=True)[name].transform("mean")
        anom = (df[name] - clim).to_numpy()
        wide = pd.DataFrame({"adm2": df["adm2"].astype(str), "time": df["time"], "a": anom}) \
            .pivot(index="time", columns="adm2", values="a")[order]
        c = wide.corr().to_numpy()
        ax.scatter(d[iu], c[iu], s=6, alpha=.5)
        ax.set_title(LABELS[name]); ax.set_xlabel("jarak (km)"); ax.grid(alpha=.3)
        near, far = c[iu][d[iu] < 50], c[iu][d[iu] > 150]
        summary[name] = dict(r_dekat_lt50km=float(near.mean()), r_jauh_gt150km=float(far.mean()))
    axes[0].set_ylabel("korelasi anomali")
    savefig(fig, "korelasi_spasial")
    return summary


def annual_trend(df):
    local = df["time"] + WIB
    full_years = df[local.dt.year < pd.Timestamp.now().year]
    t = full_years.groupby((full_years["time"] + WIB).dt.year)[TARGET_NAMES].mean()
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for ax, name in zip(axes, TARGET_NAMES):
        ax.plot(t.index, t[name], marker="o")
        ax.set_title(f"{LABELS[name]} — rata-rata tahunan"); ax.grid(alpha=.3)
    savefig(fig, "tren_tahunan")
    return t


def plot_map(locations):
    coast = coast_distances(locations["lat"], locations["lon"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sc = ax.scatter(locations["lon"], locations["lat"], c=locations["elevasi"], cmap="terrain", s=60,
                    edgecolor="k", vmin=0, vmax=900)
    for r in locations.itertuples():
        ax.annotate(r.ibukota, (r.lon, r.lat), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, label="elevasi (m)")
    ax.set_xlabel("bujur"); ax.set_ylabel("lintang"); ax.set_title("35 titik lokasi (pusat pemerintahan)")
    ax.grid(alpha=.3)
    savefig(fig, "peta_lokasi")
    return coast


def main():
    df = pd.read_parquet(settings.DATA_DIR / "processed" / "hourly.parquet")
    df["adm2"] = df["adm2"].astype(str)
    locations = pd.read_csv(settings.LOCATIONS_CSV, dtype={"adm2": str, "adm4_bmkg": str})

    comp, expected = completeness(df, locations)
    stats, rain = target_stats(df)
    plot_distributions(df)
    plot_cycles(df)
    acf = autocorrelation(df)
    spatial = spatial_correlation(df, locations)
    trend = annual_trend(df)
    plot_map(locations)

    lines = [
        "# Ringkasan EDA",
        "",
        f"- Periode: {df['time'].min()} – {df['time'].max()} UTC ({len(expected):,} jam)",
        f"- Lokasi: {df['adm2'].nunique()}; total baris: {len(df):,}",
        f"- Jam hilang total: {int(comp['jam_hilang'].sum())}; nilai kosong total: {int(comp['nilai_kosong'].sum())}",
        "",
        "## Statistik target",
        "",
        stats.round(2).to_markdown(),
        "",
        "## Curah hujan",
        "",
        f"- Proporsi jam tanpa hujan (= 0 mm): {rain['nol']:.1%}",
        f"- Proporsi jam ≥ 0,1 mm: {rain[0.1]:.1%}; ≥ 1 mm: {rain[1.0]:.1%}; ≥ 10 mm: {rain[10.0]:.2%}",
        "",
        "## Autokorelasi (rata-rata 35 lokasi)",
        "",
        pd.DataFrame(acf).T.round(3).to_markdown(),
        "",
        "## Korelasi spasial anomali",
        "",
        pd.DataFrame(spatial).T.round(3).to_markdown(),
        "",
        "## Rata-rata tahunan",
        "",
        trend.round(2).rename_axis("tahun").to_markdown(),
        "",
        "## Kelengkapan per lokasi",
        "",
        comp.to_markdown(index=False, disable_numparse=True),
    ]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
