"""Bangun config/locations.csv: 35 kabupaten/kota Jawa Tengah.

Langkah:
1. Geocoding pusat pemerintahan via Nominatim (OSM).
2. Reverse geocoding -> nama kelurahan/kecamatan -> kode adm4 (Permendagri).
3. Verifikasi adm4 ke API BMKG (koordinat kelurahan harus dekat titik).
4. Elevasi dan sel grid ECMWF IFS dari Open-Meteo.

Sekali jalan; hasilnya di-commit. Kode wilayah: data/kodewilayah.csv
(github.com/kodewilayah/permendagri-72-2019).
"""

import csv
import math
import re
import sys
import time
from pathlib import Path

import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "WhoCry-skripsi/0.1 (github.com/SirojMun/akuKehujanan)"}
CACHE = ROOT / "data" / "locations_cache.json"

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=6, connect=6, read=6, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504))))

# adm2: (nama, ibu kota, perkiraan lat, lon) -- perkiraan dipakai untuk cek kewajaran.
REGENCIES = {
    "33.01": ("Kabupaten Cilacap", "Cilacap", -7.727, 109.009),
    "33.02": ("Kabupaten Banyumas", "Purwokerto", -7.424, 109.234),
    "33.03": ("Kabupaten Purbalingga", "Purbalingga", -7.388, 109.364),
    "33.04": ("Kabupaten Banjarnegara", "Banjarnegara", -7.397, 109.698),
    "33.05": ("Kabupaten Kebumen", "Kebumen", -7.668, 109.652),
    "33.06": ("Kabupaten Purworejo", "Purworejo", -7.713, 110.009),
    "33.07": ("Kabupaten Wonosobo", "Wonosobo", -7.361, 109.903),
    "33.08": ("Kabupaten Magelang", "Mungkid", -7.584, 110.267),
    "33.09": ("Kabupaten Boyolali", "Boyolali", -7.533, 110.596),
    "33.10": ("Kabupaten Klaten", "Klaten", -7.706, 110.606),
    "33.11": ("Kabupaten Sukoharjo", "Sukoharjo", -7.682, 110.840),
    "33.12": ("Kabupaten Wonogiri", "Wonogiri", -7.814, 110.926),
    "33.13": ("Kabupaten Karanganyar", "Karanganyar", -7.597, 110.950),
    "33.14": ("Kabupaten Sragen", "Sragen", -7.426, 111.022),
    "33.15": ("Kabupaten Grobogan", "Purwodadi", -7.087, 110.915),
    "33.16": ("Kabupaten Blora", "Blora", -6.969, 111.418),
    "33.17": ("Kabupaten Rembang", "Rembang", -6.707, 111.341),
    "33.18": ("Kabupaten Pati", "Pati", -6.755, 111.038),
    "33.19": ("Kabupaten Kudus", "Kudus", -6.805, 110.841),
    "33.20": ("Kabupaten Jepara", "Jepara", -6.589, 110.668),
    "33.21": ("Kabupaten Demak", "Demak", -6.894, 110.638),
    "33.22": ("Kabupaten Semarang", "Ungaran", -7.139, 110.405),
    "33.23": ("Kabupaten Temanggung", "Temanggung", -7.316, 110.175),
    "33.24": ("Kabupaten Kendal", "Kendal", -6.920, 110.203),
    "33.25": ("Kabupaten Batang", "Batang", -6.909, 109.730),
    "33.26": ("Kabupaten Pekalongan", "Kajen", -7.025, 109.588),
    "33.27": ("Kabupaten Pemalang", "Pemalang", -6.890, 109.381),
    "33.28": ("Kabupaten Tegal", "Slawi", -6.983, 109.135),
    "33.29": ("Kabupaten Brebes", "Brebes", -6.871, 109.040),
    "33.71": ("Kota Magelang", "Magelang", -7.470, 110.218),
    "33.72": ("Kota Surakarta", "Surakarta", -7.567, 110.828),
    "33.73": ("Kota Salatiga", "Salatiga", -7.331, 110.493),
    "33.74": ("Kota Semarang", "Semarang", -6.982, 110.409),
    "33.75": ("Kota Pekalongan", "Pekalongan", -6.889, 109.675),
    "33.76": ("Kota Tegal", "Tegal", -6.869, 109.140),
}


def haversine_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def norm(name):
    name = name.lower()
    name = re.sub(r"\b(kelurahan|desa|kecamatan|kel\.|kec\.)\b", "", name)
    return re.sub(r"[^a-z0-9]", "", name)


def load_codes():
    codes = {}
    with open(ROOT / "data" / "kodewilayah.csv", newline="", encoding="utf-8") as f:
        for code, name in csv.reader(f):
            if code.startswith("33."):
                codes[code] = name
    return codes


def nominatim(path, **params):
    time.sleep(1.1)  # kebijakan Nominatim: maks 1 req/detik
    params.update(format="jsonv2", addressdetails=1)
    r = session.get(f"https://nominatim.openstreetmap.org/{path}", params=params,
                     headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def geocode(ibukota, approx_lat, approx_lon):
    results = nominatim("search", q=f"{ibukota}, Jawa Tengah, Indonesia",
                        countrycodes="id", limit=10)
    places = [x for x in results if x.get("category") == "place"
              or x.get("class") == "place" or x.get("type") == "administrative"]
    best = min(places or results,
               key=lambda x: haversine_km(float(x["lat"]), float(x["lon"]), approx_lat, approx_lon))
    return float(best["lat"]), float(best["lon"])


def bmkg_point(adm4):
    time.sleep(1.1)  # BMKG: 60 req/menit
    r = session.get("https://api.bmkg.go.id/publik/prakiraan-cuaca",
                     params={"adm4": adm4}, headers=UA, timeout=30, allow_redirects=False)
    if r.status_code != 200:
        return None
    lok = r.json().get("lokasi", {})
    return lok.get("lat"), lok.get("lon"), lok.get("desa")


def find_adm4(adm2, ibukota, lat, lon, codes):
    addr = nominatim("reverse", lat=lat, lon=lon, zoom=18).get("address", {})
    village_names = {norm(addr[k]) for k in ("village", "suburb", "neighbourhood", "hamlet", "quarter")
                     if k in addr}
    district_names = {norm(addr[k]) for k in ("city_district", "district", "municipality", "town", "city")
                      if k in addr}
    kec = [c for c, n in codes.items() if c.startswith(adm2 + ".") and c.count(".") == 2]
    kec_match = ([c for c in kec if norm(codes[c]) in district_names]
                 or [c for c in kec if norm(ibukota) in norm(codes[c])]
                 or [c for c in kec if any(d and d in norm(codes[c]) for d in district_names)])
    desa_all = [c for c in codes if c.count(".") == 3 and c[:8] in kec]
    exact = [c for c in desa_all if norm(codes[c]) in village_names
             and (not kec_match or c[:8] in kec_match)]
    candidates = exact or [c for c in desa_all if c[:8] in kec_match]

    best = None
    for c in candidates[:40]:
        pt = bmkg_point(c)
        if not pt or pt[0] is None:
            continue
        d = haversine_km(lat, lon, pt[0], pt[1])
        if best is None or d < best[1]:
            best = (c, d, pt[2])
        if exact and d < 3:
            break
    return best, addr


def openmeteo_meta(lat, lon):
    r = session.get("https://api.open-meteo.com/v1/forecast",
                     params=dict(latitude=lat, longitude=lon, hourly="temperature_2m",
                                 forecast_days=1, models="ecmwf_ifs"), timeout=30)
    r.raise_for_status()
    d = r.json()
    return d["elevation"], d["latitude"], d["longitude"]


def main():
    codes = load_codes()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    rows = []
    for adm2, (nama, ibukota, alat, alon) in REGENCIES.items():
        if adm2 in cache and cache[adm2]["adm4_bmkg"]:
            rows.append(cache[adm2])
            continue
        lat, lon = geocode(ibukota, alat, alon)
        off = haversine_km(lat, lon, alat, alon)
        if off > 8:  # geocoding meleset jauh -> pakai perkiraan
            print(f"!! {nama}: geocode meleset {off:.1f} km, pakai perkiraan", file=sys.stderr)
            lat, lon = alat, alon
        best, addr = find_adm4(adm2, ibukota, lat, lon, codes)
        elev, glat, glon = openmeteo_meta(lat, lon)
        adm4, dist, desa = best if best else ("", float("nan"), "")
        print(f"{adm2} {nama:24s} {lat:.4f},{lon:.4f} adm4={adm4} ({desa}, {dist:.1f} km) "
              f"grid={glat:.3f},{glon:.3f} elev={elev}")
        rows.append(dict(adm2=adm2, nama=nama, ibukota=ibukota, lat=round(lat, 5), lon=round(lon, 5),
                         elevasi=elev, adm4_bmkg=adm4, desa_bmkg=desa,
                         jarak_bmkg_km=round(dist, 2), grid_lat=glat, grid_lon=glon))
        cache[adm2] = rows[-1]
        CACHE.write_text(json.dumps(cache, indent=1))

    out = ROOT / "config" / "locations.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"tersimpan: {out}")


if __name__ == "__main__":
    main()
