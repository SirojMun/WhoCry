"""Bangun config/jateng.geojson: batas 35 kabupaten/kota Jawa Tengah untuk peta dasbor.

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


def rounded(c):
    return [rounded(x) for x in c] if isinstance(c[0], list) else [round(c[0], 4), round(c[1], 4)]


def main():
    src = requests.get(URL, timeout=300).json()
    loc = pd.read_csv(settings.LOCATIONS_CSV, dtype={"adm2": str})
    by_name = {n.replace("Kabupaten ", ""): (a, n) for a, n in zip(loc["adm2"], loc["nama"])}
    feats = []
    for f in src["features"]:
        name = f["properties"]["shapeName"]
        if name not in by_name and name not in NON_ADMIN:
            continue
        adm2, nama = by_name.get(name, (None, name))
        feats.append({"type": "Feature", "properties": {"adm2": adm2, "nama": nama},
                      "geometry": {"type": f["geometry"]["type"],
                                   "coordinates": rounded(f["geometry"]["coordinates"])}})
    matched = {f["properties"]["adm2"] for f in feats} - {None}
    assert matched == set(loc["adm2"]), f"tidak cocok: {set(loc['adm2']) - matched}"
    out = settings.ROOT / "config" / "jateng.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
    print(f"{out}: {len(feats)} fitur, {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
