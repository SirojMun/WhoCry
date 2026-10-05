"""Peringatan cuaca ke Telegram berdasarkan ambang pada prediksi 24 jam ke depan.

Setiap kejadian (lokasi, variabel, tanggal WIB) hanya diperingatkan sekali;
riwayatnya disimpan di file state JSON agar tidak terkirim ulang setiap jam.
"""

import json
import os

import pandas as pd
import requests

WIB = pd.Timedelta(hours=7)

# variabel: (operator, ambang, label, satuan)
THRESHOLDS = {
    "precipitation": (">=", 10.0, "Hujan lebat", "mm/jam"),
    "temperature_2m": (">=", 35.0, "Suhu ekstrem", "°C"),
    "relative_humidity_2m": ("<=", 40.0, "Udara sangat kering", "%"),
}


def find_events(pred, locations):
    """Satu baris per (lokasi, variabel, tanggal WIB) yang melewati ambang; diambil nilai terekstrem."""
    names = locations.set_index("adm2")["nama"]
    events = []
    for var, (op, thr, label, unit) in THRESHOLDS.items():
        hit = pred[pred[var] >= thr] if op == ">=" else pred[pred[var] <= thr]
        if hit.empty:
            continue
        hit = hit.assign(tanggal=(hit["valid_time"] + WIB).dt.date)
        for (adm2, tanggal), g in hit.groupby(["adm2", "tanggal"]):
            row = g.loc[g[var].idxmax() if op == ">=" else g[var].idxmin()]
            events.append(dict(key=f"{adm2}|{var}|{tanggal}", adm2=adm2, nama=names.get(adm2, adm2),
                               label=label, value=float(row[var]), unit=unit,
                               waktu_wib=(row["valid_time"] + WIB).strftime("%d/%m %H:%M")))
    return events


def format_message(events, model_name):
    lines = ["⚠️ *Peringatan cuaca — prediksi 24 jam ke depan*", ""]
    for e in sorted(events, key=lambda e: (e["label"], e["nama"])):
        lines.append(f"• {e['label']}: *{e['nama']}* — {e['value']:.1f} {e['unit']} ({e['waktu_wib']} WIB)")
    lines += ["", f"_Model: {model_name}. Data: Open-Meteo (CC-BY 4.0)._"]
    return "\n".join(lines)


def send_telegram(text, token=None, chat_id=None):
    token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json=dict(chat_id=chat_id, text=text, parse_mode="Markdown"), timeout=30)
    r.raise_for_status()
    return True


def process(pred, locations, state_path):
    """Cari kejadian baru, kirim (jika kredensial ada), perbarui state. Mengembalikan kejadian baru."""
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    new = [e for e in find_events(pred, locations) if e["key"] not in state]
    if new:
        sent = send_telegram(format_message(new, pred["model"].iloc[0]))
        if sent:
            for e in new:
                state[e["key"]] = pd.Timestamp.now(tz="UTC").isoformat()
    # Buang state lebih dari 7 hari agar file tetap kecil.
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    state = {k: v for k, v in state.items() if pd.Timestamp(v) > cutoff}
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=1))
    return new
