# Tugas Besar Metode Numerik: Perbandingan Metode Penyelesaian SPL

Kelompok 9 — Michael Alexander Newton, Arif Mufti Tharsa, Razqa Azaki.
Dosen Pengampu: Dr. Muhammad Zaki Almuzakki, S.Si., M.Si., M.Sc.
Universitas Pertamina, Semester 4, 2026.

Membandingkan waktu eksekusi, resource, dan akurasi empat metode
penyelesaian SPL (Eliminasi Gauss, Gauss-Jordan, Jacobi, Gauss-Seidel).
Laporan lengkap ada di folder `reports/`.

## Struktur

- `src/solvers.py` — 4 fungsi solver SPL
- `src/metrics.py` — residual error dan R-squared
- `src/run_experiment.py` — loop eksperimen penuh 
- `src/visualisasi.py` — pembuatan grafik dari data eksperimen
- `data/hasil_eksperimen.csv` — hasil lengkap
- `reports/` — PDF laporan lengkap

## Menjalankan

```bash
pip install -r requirements.txt
python -c "from src.solvers import eliminasi_gauss_murni; print(eliminasi_gauss_murni([[10,1,1],[1,12,1],[1,1,15]], [12,14,17]))"
```