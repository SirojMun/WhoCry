# Ringkasan EDA

- Periode: 2017-01-01 00:00:00 – 2026-10-05 23:00:00 UTC (85,560 jam)
- Lokasi: 35; total baris: 2,994,600
- Jam hilang total: 0; nilai kosong total: 0

## Statistik target

|                      |      count |   mean |   std |   min |   1% |   50% |   99% |   max |
|:---------------------|-----------:|-------:|------:|------:|-----:|------:|------:|------:|
| temperature_2m       | 2.9946e+06 |  26    |  3.1  |  12.1 | 19.2 |  25.6 |  33.9 |  40.1 |
| relative_humidity_2m | 2.9946e+06 |  80.41 | 14.74 |  12   | 37   |  84   |  99   | 100   |
| cloud_cover          | 2.9946e+06 |  75.61 | 32.25 |   0   |  0   |  94   | 100   | 100   |
| precipitation        | 2.9946e+06 |   0.3  |  1.14 |   0   |  0   |   0   |   5.4 |  67.3 |

## Curah hujan

- Proporsi jam tanpa hujan (= 0 mm): 67.5%
- Proporsi jam ≥ 0,1 mm: 32.5%; ≥ 1 mm: 7.2%; ≥ 10 mm: 0.26%

## Autokorelasi (rata-rata 35 lokasi)

|                      |   lag1 |   lag3 |   lag6 |   lag12 |   lag24 |
|:---------------------|-------:|-------:|-------:|--------:|--------:|
| temperature_2m       |  0.939 |  0.639 |  0.043 |  -0.495 |   0.916 |
| relative_humidity_2m |  0.944 |  0.718 |  0.303 |  -0.076 |   0.862 |
| cloud_cover          |  0.844 |  0.652 |  0.512 |   0.405 |   0.404 |
| precipitation        |  0.471 |  0.214 |  0.085 |   0.038 |   0.168 |

## Korelasi spasial anomali

|                      |   r_dekat_lt50km |   r_jauh_gt150km |
|:---------------------|-----------------:|-----------------:|
| temperature_2m       |            0.724 |            0.489 |
| relative_humidity_2m |            0.68  |            0.488 |
| cloud_cover          |            0.766 |            0.512 |
| precipitation        |            0.289 |            0.089 |

## Rata-rata tahunan

|   tahun |   temperature_2m |   relative_humidity_2m |   cloud_cover |   precipitation |
|--------:|-----------------:|-----------------------:|--------------:|----------------:|
|    2017 |            25.88 |                  80.53 |         77.81 |            0.27 |
|    2018 |            25.96 |                  77.76 |         71.63 |            0.24 |
|    2019 |            26.21 |                  77.43 |         69.26 |            0.26 |
|    2020 |            26.09 |                  82.44 |         75.63 |            0.37 |
|    2021 |            25.86 |                  82.38 |         76.89 |            0.35 |
|    2022 |            25.53 |                  83.87 |         86.16 |            0.37 |
|    2023 |            26.17 |                  77.84 |         73.07 |            0.23 |
|    2024 |            26.39 |                  80.18 |         77.24 |            0.29 |
|    2025 |            25.87 |                  82.46 |         81.48 |            0.33 |

## Kelengkapan per lokasi

| adm2   | jam   | jam_hilang   | nilai_kosong   | nama                   |
|:-------|:------|:-------------|:---------------|:-----------------------|
| 33.01  | 85560 | 0            | 0              | Kabupaten Cilacap      |
| 33.02  | 85560 | 0            | 0              | Kabupaten Banyumas     |
| 33.03  | 85560 | 0            | 0              | Kabupaten Purbalingga  |
| 33.04  | 85560 | 0            | 0              | Kabupaten Banjarnegara |
| 33.05  | 85560 | 0            | 0              | Kabupaten Kebumen      |
| 33.06  | 85560 | 0            | 0              | Kabupaten Purworejo    |
| 33.07  | 85560 | 0            | 0              | Kabupaten Wonosobo     |
| 33.08  | 85560 | 0            | 0              | Kabupaten Magelang     |
| 33.09  | 85560 | 0            | 0              | Kabupaten Boyolali     |
| 33.10  | 85560 | 0            | 0              | Kabupaten Klaten       |
| 33.11  | 85560 | 0            | 0              | Kabupaten Sukoharjo    |
| 33.12  | 85560 | 0            | 0              | Kabupaten Wonogiri     |
| 33.13  | 85560 | 0            | 0              | Kabupaten Karanganyar  |
| 33.14  | 85560 | 0            | 0              | Kabupaten Sragen       |
| 33.15  | 85560 | 0            | 0              | Kabupaten Grobogan     |
| 33.16  | 85560 | 0            | 0              | Kabupaten Blora        |
| 33.17  | 85560 | 0            | 0              | Kabupaten Rembang      |
| 33.18  | 85560 | 0            | 0              | Kabupaten Pati         |
| 33.19  | 85560 | 0            | 0              | Kabupaten Kudus        |
| 33.20  | 85560 | 0            | 0              | Kabupaten Jepara       |
| 33.21  | 85560 | 0            | 0              | Kabupaten Demak        |
| 33.22  | 85560 | 0            | 0              | Kabupaten Semarang     |
| 33.23  | 85560 | 0            | 0              | Kabupaten Temanggung   |
| 33.24  | 85560 | 0            | 0              | Kabupaten Kendal       |
| 33.25  | 85560 | 0            | 0              | Kabupaten Batang       |
| 33.26  | 85560 | 0            | 0              | Kabupaten Pekalongan   |
| 33.27  | 85560 | 0            | 0              | Kabupaten Pemalang     |
| 33.28  | 85560 | 0            | 0              | Kabupaten Tegal        |
| 33.29  | 85560 | 0            | 0              | Kabupaten Brebes       |
| 33.71  | 85560 | 0            | 0              | Kota Magelang          |
| 33.72  | 85560 | 0            | 0              | Kota Surakarta         |
| 33.73  | 85560 | 0            | 0              | Kota Salatiga          |
| 33.74  | 85560 | 0            | 0              | Kota Semarang          |
| 33.75  | 85560 | 0            | 0              | Kota Pekalongan        |
| 33.76  | 85560 | 0            | 0              | Kota Tegal             |