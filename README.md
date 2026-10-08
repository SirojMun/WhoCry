# WhoCry

Prediksi cuaca per jam (suhu, kelembapan, tutupan awan, curah hujan) untuk
40 kabupaten/kota di Jawa Tengah dan DIY menggunakan regresi linier, LSTM, dan GRU
dengan fitur spasial, disertai pelatihan ulang berkala dan sistem pemantauan *realtime*.

## Struktur

```
src/whocry/   kode utama (pengambilan data, fitur, model, pelatihan)
scripts/      skrip yang dijalankan langsung (unduh data, kolektor, dll.)
kaggle/       runner pelatihan di Kaggle (GPU): `kaggle kernels push -p kaggle`
tests/        pengujian otomatis
data/         data lokal (tidak di-push)
```

## Instalasi

```bash
pip install -r requirements.txt
pip install -e .   # opsional, agar paket whocry bisa di-import
```

PyTorch sudah tersedia di Kaggle. Untuk lokal, pasang sesuai petunjuk di
https://pytorch.org.

## Sumber Data

- Data cuaca: [Open-Meteo](https://open-meteo.com) (CC-BY 4.0)
- Prakiraan pembanding: [BMKG](https://data.bmkg.go.id/prakiraan-cuaca)
