"""Bangun config/wilayah.geojson: batas kabupaten/kota Jawa Tengah & DIY untuk peta dasbor.

Sumber: geoBoundaries IDN ADM2 (BPS/OCHA, CC BY 3.0 IGO), versi simplified.

    python scripts/build_geojson.py
"""

import json

import pandas as pd
import requests

from whocry import settings

URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IDN/ADM2/"
       "geoBoundaries-IDN-ADM2_simplified.geojson")
NON_ADMIN = {"Hutan", "Wadung Kedungombo"}  # unit khusus di dalam Jateng; digambar abu-abu
ALIASES = {"Gunung Kidul": "Gunungkidul"}  # ejaan geoBoundaries -> ejaan resmi


def rounded(c):
    return [rounded(x) for x in c] if isinstance(c[0], list) else [round(c[0], 4), round(c[1], 4)]


def signed_area(ring):
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:]))


def d3_winding(polygon):
    """d3-geo (Vega/Altair) butuh ring luar searah jarum jam dan lubang berlawanan,
    kebalikan dari RFC 7946. Tanpa ini setiap poligon digambar sebagai 'seluruh bumi'."""
    return [r if (signed_area(r) < 0) == (i == 0) else r[::-1] for i, r in enumerate(polygon)]


ISLAND_LAT = -6.2  # bagian poligon di utara garis ini (Kep. Karimunjawa, Jepara) dibuang agar peta tidak memanjang


def fix_geometry(g):
    coords = rounded(g["coordinates"])
    if g["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": d3_winding(coords)}
    parts = [d3_winding(p) for p in coords if min(y for _, y in p[0]) < ISLAND_LAT]
    return {"type": "MultiPolygon", "coordinates": parts}


def main():
    src = requests.get(URL, timeout=300).json()
    loc = pd.read_csv(settings.LOCATIONS_CSV, dtype={"adm2": str})
    by_name = {n.replace("Kabupaten ", ""): (a, n) for a, n in zip(loc["adm2"], loc["nama"])}
    feats = []
    for f in src["features"]:
        name = ALIASES.get(f["properties"]["shapeName"], f["properties"]["shapeName"])
        if name not in by_name and name not in NON_ADMIN:
            continue
        adm2, nama = by_name.get(name, (None, name))
        feats.append({"type": "Feature", "properties": {"adm2": adm2, "nama": nama},
                      "geometry": fix_geometry(f["geometry"])})
    matched = {f["properties"]["adm2"] for f in feats} - {None}
    assert matched == set(loc["adm2"]), f"tidak cocok: {set(loc['adm2']) - matched}"
    for f in feats:  # cek: semua ring luar searah jarum jam (luas bertanda negatif)
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        assert all(signed_area(p[0]) < 0 for p in polys), f["properties"]["nama"]
    out = settings.ROOT / "config" / "wilayah.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
    print(f"{out}: {len(feats)} fitur, {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
