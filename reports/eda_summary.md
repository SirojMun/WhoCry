# Ringkasan EDA

- Periode: 2017-01-01 00:00:00 – 2026-10-07 23:00:00 UTC (85,608 jam)
- Lokasi: 40; total baris: 3,424,320
- Jam hilang total: 0; nilai kosong total: 0

## Statistik target

|                      |       count |   mean |   std |   min |   1% |   50% |   99% |   max |
|:---------------------|------------:|-------:|------:|------:|-----:|------:|------:|------:|
| temperature_2m       | 3.42432e+06 |  25.94 |  3.04 |  12.1 | 19.3 |  25.5 |  33.8 |  40.1 |
| relative_humidity_2m | 3.42432e+06 |  80.73 | 14.47 |  12   | 37   |  84   |  99   | 100   |
| cloud_cover          | 3.42432e+06 |  75.62 | 32.18 |   0   |  0   |  94   | 100   | 100   |
| precipitation        | 3.42432e+06 |   0.29 |  1.13 |   0   |  0   |   0   |   5.4 |  67.3 |

## Curah hujan

- Proporsi jam tanpa hujan (= 0 mm): 67.2%
- Proporsi jam ≥ 0,1 mm: 32.8%; ≥ 1 mm: 7.2%; ≥ 10 mm: 0.25%

## Autokorelasi (rata-rata 40 lokasi)

|                      |   lag1 |   lag3 |   lag6 |   lag12 |   lag24 |
|:---------------------|-------:|-------:|-------:|--------:|--------:|
| temperature_2m       |  0.939 |  0.638 |  0.041 |  -0.495 |   0.916 |
| relative_humidity_2m |  0.943 |  0.712 |  0.287 |  -0.105 |   0.863 |
| cloud_cover          |  0.843 |  0.651 |  0.509 |   0.403 |   0.404 |
| precipitation        |  0.473 |  0.216 |  0.086 |   0.039 |   0.165 |

## Korelasi spasial anomali

|                      |   r_dekat_lt50km |   r_jauh_gt150km |
|:---------------------|-----------------:|-----------------:|
| temperature_2m       |            0.719 |            0.48  |
| relative_humidity_2m |            0.674 |            0.477 |
| cloud_cover          |            0.759 |            0.509 |
| precipitation        |            0.292 |            0.088 |

## Rata-rata tahunan

|   tahun |   temperature_2m |   relative_humidity_2m |   cloud_cover |   precipitation |
|--------:|-----------------:|-----------------------:|--------------:|----------------:|
|    2017 |            25.81 |                  80.92 |         77.89 |            0.27 |
|    2018 |            25.89 |                  78.19 |         71.65 |            0.24 |
|    2019 |            26.13 |                  77.83 |         69.21 |            0.26 |
|    2020 |            26.05 |                  82.68 |         75.54 |            0.36 |
|    2021 |            25.83 |                  82.61 |         76.85 |            0.34 |
|    2022 |            25.49 |                  84.09 |         86.19 |            0.37 |
|    2023 |            26.09 |                  78.29 |         73.18 |            0.22 |
|    2024 |            26.33 |                  80.51 |         77.24 |            0.29 |
|    2025 |            25.83 |                  82.69 |         81.5  |            0.33 |

## Kelengkapan per lokasi

| adm2   | jam   | jam_hilang   | nilai_kosong   | nama                   |
|:-------|:------|:-------------|:---------------|:-----------------------|
| 33.01  | 85608 | 0            | 0              | Kabupaten Cilacap      |
| 33.02  | 85608 | 0            | 0              | Kabupaten Banyumas     |
| 33.03  | 85608 | 0            | 0              | Kabupaten Purbalingga  |
| 33.04  | 85608 | 0            | 0              | Kabupaten Banjarnegara |
| 33.05  | 85608 | 0            | 0              | Kabupaten Kebumen      |
| 33.06  | 85608 | 0            | 0              | Kabupaten Purworejo    |
| 33.07  | 85608 | 0            | 0              | Kabupaten Wonosobo     |
| 33.08  | 85608 | 0            | 0              | Kabupaten Magelang     |
| 33.09  | 85608 | 0            | 0              | Kabupaten Boyolali     |
| 33.10  | 85608 | 0            | 0              | Kabupaten Klaten       |
| 33.11  | 85608 | 0            | 0              | Kabupaten Sukoharjo    |
| 33.12  | 85608 | 0            | 0              | Kabupaten Wonogiri     |
| 33.13  | 85608 | 0            | 0              | Kabupaten Karanganyar  |
| 33.14  | 85608 | 0            | 0              | Kabupaten Sragen       |
| 33.15  | 85608 | 0            | 0              | Kabupaten Grobogan     |
| 33.16  | 85608 | 0            | 0              | Kabupaten Blora        |
| 33.17  | 85608 | 0            | 0              | Kabupaten Rembang      |
| 33.18  | 85608 | 0            | 0              | Kabupaten Pati         |
| 33.19  | 85608 | 0            | 0              | Kabupaten Kudus        |
| 33.20  | 85608 | 0            | 0              | Kabupaten Jepara       |
| 33.21  | 85608 | 0            | 0              | Kabupaten Demak        |
| 33.22  | 85608 | 0            | 0              | Kabupaten Semarang     |
| 33.23  | 85608 | 0            | 0              | Kabupaten Temanggung   |
| 33.24  | 85608 | 0            | 0              | Kabupaten Kendal       |
| 33.25  | 85608 | 0            | 0              | Kabupaten Batang       |
| 33.26  | 85608 | 0            | 0              | Kabupaten Pekalongan   |
| 33.27  | 85608 | 0            | 0              | Kabupaten Pemalang     |
| 33.28  | 85608 | 0            | 0              | Kabupaten Tegal        |
| 33.29  | 85608 | 0            | 0              | Kabupaten Brebes       |
| 33.71  | 85608 | 0            | 0              | Kota Magelang          |
| 33.72  | 85608 | 0            | 0              | Kota Surakarta         |
| 33.73  | 85608 | 0            | 0              | Kota Salatiga          |
| 33.74  | 85608 | 0            | 0              | Kota Semarang          |
| 33.75  | 85608 | 0            | 0              | Kota Pekalongan        |
| 33.76  | 85608 | 0            | 0              | Kota Tegal             |
| 34.01  | 85608 | 0            | 0              | Kabupaten Kulon Progo  |
| 34.02  | 85608 | 0            | 0              | Kabupaten Bantul       |
| 34.03  | 85608 | 0            | 0              | Kabupaten Gunungkidul  |
| 34.04  | 85608 | 0            | 0              | Kabupaten Sleman       |
| 34.71  | 85608 | 0            | 0              | Kota Yogyakarta        |