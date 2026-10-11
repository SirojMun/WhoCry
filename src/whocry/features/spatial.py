"""Fitur spasial statis dan bobot tetangga antarlokasi."""

import numpy as np

EARTH_DIAMETER_KM = 12742.0

# Titik-titik garis pantai yang disederhanakan (lat, lon), dari barat ke timur.
NORTH_COAST = [  # Laut Jawa
    (-6.80, 108.83), (-6.84, 109.03), (-6.85, 109.14), (-6.83, 109.40), (-6.86, 109.68),
    (-6.88, 109.75), (-6.90, 110.15), (-6.95, 110.40), (-6.84, 110.53), (-6.59, 110.65),
    (-6.42, 110.90), (-6.67, 111.15), (-6.69, 111.35), (-6.65, 111.45), (-6.73, 111.69),
]
SOUTH_COAST = [  # Samudra Hindia
    (-7.70, 108.85), (-7.74, 109.02), (-7.70, 109.15), (-7.73, 109.40), (-7.80, 109.60),
    (-7.88, 109.90), (-7.90, 110.05), (-7.95, 110.15), (-8.02, 110.33), (-8.13, 110.60),
    (-8.20, 110.85),
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Jarak lingkaran besar; menerima skalar maupun array (broadcast)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return EARTH_DIAMETER_KM * np.arcsin(np.sqrt(a))


def _densify(points, steps=50):
    pts = np.asarray(points)
    segs = [np.linspace(a, b, steps, endpoint=False) for a, b in zip(pts[:-1], pts[1:])]
    return np.vstack(segs + [pts[-1:]])


def coast_distances(lat, lon):
    """Jarak (km) setiap lokasi ke pantai utara dan pantai selatan. Keluaran [L, 2]."""
    out = []
    for coast in (NORTH_COAST, SOUTH_COAST):
        c = _densify(coast)
        d = haversine_km(np.asarray(lat)[:, None], np.asarray(lon)[:, None], c[None, :, 0], c[None, :, 1])
        out.append(d.min(axis=1))
    return np.stack(out, axis=1)


def static_features(locations):
    """Fitur statis S1: lat, lon, elevasi, jarak ke pantai utara, jarak ke pantai selatan."""
    coast = coast_distances(locations["lat"].to_numpy(), locations["lon"].to_numpy())
    return np.column_stack([
        locations["lat"].to_numpy(), locations["lon"].to_numpy(),
        locations["elevasi"].to_numpy(), coast,
    ]).astype("float32")


STATIC_NAMES = ["lat", "lon", "elevasi", "jarak_pantai_utara", "jarak_pantai_selatan"]


def idw_weights(lat, lon, k=4, power=2):
    """Matriks bobot [L, L]: baris i berisi bobot IDW dari k tetangga terdekat (tanpa diri sendiri)."""
    lat, lon = np.asarray(lat), np.asarray(lon)
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    np.fill_diagonal(d, np.inf)
    w = np.zeros_like(d)
    nearest = np.argsort(d, axis=1)[:, :k]
    rows = np.arange(len(lat))[:, None]
    w[rows, nearest] = 1.0 / np.maximum(d[rows, nearest], 1.0) ** power
    return (w / w.sum(axis=1, keepdims=True)).astype("float32")


UPWIND_RADIUS_KM = 150


def upwind_geometry(lat, lon, radius_km=UPWIND_RADIUS_KM):
    """Geometri untuk bobot upwind: ex/ey[i, j] = vektor satuan dari tetangga j MENUJU lokasi i
    (timur, utara), inv_d[i, j] = 1/jarak untuk tetangga dalam radius (0 untuk diri sendiri/di luar radius)."""
    lat, lon = np.asarray(lat, dtype="float64"), np.asarray(lon, dtype="float64")
    dx = (lon[:, None] - lon[None, :]) * 111.32 * np.cos(np.radians(lat))[:, None]
    dy = (lat[:, None] - lat[None, :]) * 110.57
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    norm = np.where(d > 0, np.hypot(dx, dy), 1.0)
    inv_d = np.where((d > 0) & (d < radius_km), 1.0 / np.maximum(d, 1.0), 0.0)
    return (dx / norm).astype("float32"), (dy / norm).astype("float32"), inv_d.astype("float32")


def upwind_weights(u, v, ex, ey, inv_d, fallback):
    """Bobot tetangga berdasarkan arah angin di lokasi target.

    u, v: [B, T] angin (vektor tiupan) di lokasi target; ex, ey, inv_d, fallback: [B, L] baris geometri
    untuk lokasi target. Tetangga j berbobot cos(sudut antara angin dan arah j->i)^2 / jarak, hanya jika
    angin bertiup dari j ke i. Bila tidak ada tetangga yang searah angin, dipakai bobot IDW (fallback).
    Keluaran [B, T, L], tiap baris berjumlah 1.
    """
    spd = np.hypot(u, v)[..., None] + 1e-6
    cos = (u[..., None] * ex[:, None, :] + v[..., None] * ey[:, None, :]) / spd
    w = np.clip(cos, 0, None) ** 2 * inv_d[:, None, :]
    s = w.sum(-1, keepdims=True)
    return np.where(s > 0, w / np.where(s > 0, s, 1.0), fallback[:, None, :]).astype("float32")
