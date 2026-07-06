import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc  # Indikator Memori bawaan Python

from solvers import (
    eliminasi_gauss_murni,
    eliminasi_gauss_jordan_murni,
    iterasi_jacobi_murni,
    iterasi_gauss_seidel_murni,
)
from metrics import hitung_residual_error, hitung_r2


# ========================================================
# 2. PROSES REKAYASA DATA & PENGUKURAN INDIKATOR NYATA
# ========================================================
skala_tugas = [10] + list(range(100, 2001, 100))

waktu_gauss, waktu_gj, waktu_jacobi, waktu_gs = [], [], [], []
mem_gauss, mem_gj, mem_jacobi, mem_gs = [], [], [], []
err_gauss, err_gj, err_jacobi, err_gs = [], [], [], []
it_jacobi, it_gs = [], []

print("⚡ MEMULAI PENGUJIAN MULTI-INDIKATOR PADA HARDWARE LAPTOP ANDA ⚡")
print("Catatan: Ukuran n >= 400 tanpa optimasi membutuhkan waktu ekstra berat.\n")

tracemalloc.start()

for n in skala_tugas:
    print(f"👉 Menguji Ordo Matriks {n} x {n}...")

    A_np = np.random.rand(n, n)
    np.fill_diagonal(A_np, np.sum(np.abs(A_np), axis=1) + 10)
    b_np = np.random.rand(n)
    A = A_np.tolist()
    b = b_np.tolist()

    # --- 1. Pengujian Metode Gauss ---
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    x_g, _ = eliminasi_gauss_murni(A, b)
    waktu_gauss.append(time.perf_counter() - t0)
    mem_gauss.append(tracemalloc.get_traced_memory()[1] / 1024)
    err_gauss.append(hitung_residual_error(A, x_g, b))

    # --- 2. Pengujian Metode Gauss-Jordan ---
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    x_gj, _ = eliminasi_gauss_jordan_murni(A, b)
    waktu_gj.append(time.perf_counter() - t0)
    mem_gj.append(tracemalloc.get_traced_memory()[1] / 1024)
    err_gj.append(hitung_residual_error(A, x_gj, b))

    # --- 3. Pengujian Metode Jacobi ---
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    x_jb, it_j = iterasi_jacobi_murni(A, b)
    waktu_jacobi.append(time.perf_counter() - t0)
    mem_jacobi.append(tracemalloc.get_traced_memory()[1] / 1024)
    err_jacobi.append(hitung_residual_error(A, x_jb, b))
    it_jacobi.append(it_j)

    # --- 4. Pengujian Metode Gauss-Seidel ---
    tracemalloc.reset_peak()
    t0 = time.perf_counter()
    x_gs, it_g = iterasi_gauss_seidel_murni(A, b)
    waktu_gs.append(time.perf_counter() - t0)
    mem_gs.append(tracemalloc.get_traced_memory()[1] / 1024)
    err_gs.append(hitung_residual_error(A, x_gs, b))
    it_gs.append(it_g)

tracemalloc.stop()

skala_tugas = np.array(skala_tugas)
waktu_gauss = np.array(waktu_gauss)
waktu_gj = np.array(waktu_gj)
waktu_jacobi = np.array(waktu_jacobi)
waktu_gs = np.array(waktu_gs)

# ========================================================
# 3. INDIKATOR REGRESI: POLINOMIAL DERAJAT 3 & DERAJAT 4
# ========================================================
pred3_gauss = np.polyval(np.polyfit(skala_tugas, waktu_gauss, 3), skala_tugas)
pred4_gauss = np.polyval(np.polyfit(skala_tugas, waktu_gauss, 4), skala_tugas)

pred3_gj = np.polyval(np.polyfit(skala_tugas, waktu_gj, 3), skala_tugas)
pred4_gj = np.polyval(np.polyfit(skala_tugas, waktu_gj, 4), skala_tugas)

pred3_jacobi = np.polyval(np.polyfit(skala_tugas, waktu_jacobi, 3), skala_tugas)
pred4_jacobi = np.polyval(np.polyfit(skala_tugas, waktu_jacobi, 4), skala_tugas)

pred3_gs = np.polyval(np.polyfit(skala_tugas, waktu_gs, 3), skala_tugas)
pred4_gs = np.polyval(np.polyfit(skala_tugas, waktu_gs, 4), skala_tugas)

# ========================================================
# 4. VISUALISASI GRAFIK EVALUASI POLA KENAIKAN WAKTU T(n)
# ========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

# --- Plot Subplot Derajat 3 ---
ax1.scatter(skala_tugas, waktu_gauss, color='red', alpha=0.5)
ax1.plot(skala_tugas, pred3_gauss, 'r-', label=f'Gauss (R²={hitung_r2(waktu_gauss, pred3_gauss):.4f})')
ax1.scatter(skala_tugas, waktu_gj, color='orange', alpha=0.5)
ax1.plot(skala_tugas, pred3_gj, 'o--', color='orange', label=f'Gauss-Jordan (R²={hitung_r2(waktu_gj, pred3_gj):.4f})')
ax1.scatter(skala_tugas, waktu_jacobi, color='blue', alpha=0.5)
ax1.plot(skala_tugas, pred3_jacobi, 'b-', label=f'Jacobi (R²={hitung_r2(waktu_jacobi, pred3_jacobi):.4f})')
ax1.scatter(skala_tugas, waktu_gs, color='green', alpha=0.5)
ax1.plot(skala_tugas, pred3_gs, 'g--', label=f'Gauss-Seidel (R²={hitung_r2(waktu_gs, pred3_gs):.4f})')
ax1.set_title('Pola Kenaikan Waktu: Model Regresi Derajat Tiga (3)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Ukuran Matriks ($n$)')
ax1.set_ylabel('Waktu Pemrosesan Riil (Detik)')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left')

# --- Plot Subplot Derajat 4 ---
ax2.scatter(skala_tugas, waktu_gauss, color='red', alpha=0.5)
ax2.plot(skala_tugas, pred4_gauss, 'r-', label=f'Gauss (R²={hitung_r2(waktu_gauss, pred4_gauss):.4f})')
ax2.scatter(skala_tugas, waktu_gj, color='orange', alpha=0.5)
ax2.plot(skala_tugas, pred4_gj, 'o--', color='orange', label=f'Gauss-Jordan (R²={hitung_r2(waktu_gj, pred4_gj):.4f})')
ax2.scatter(skala_tugas, waktu_jacobi, color='blue', alpha=0.5)
ax2.plot(skala_tugas, pred4_jacobi, 'b-', label=f'Jacobi (R²={hitung_r2(waktu_jacobi, pred4_jacobi):.4f})')
ax2.scatter(skala_tugas, waktu_gs, color='green', alpha=0.5)
ax2.plot(skala_tugas, pred4_gs, 'g--', label=f'Gauss-Seidel (R²={hitung_r2(waktu_gs, pred4_gs):.4f})')
ax2.set_title('Pola Kenaikan Waktu: Model Regresi Derajat Empat (4)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Ukuran Matriks ($n$)')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left')

plt.suptitle('Analisis Multi-Indikator: Hasil Validasi Pola Tren Komputasi Hardware Laptop', fontsize=12, fontweight='bold')
plt.show()

# ========================================================
# 5. RINGKASAN DATA KOMPREHENSIF UNTUK TABEL LAPORAN TUGAS
# ========================================================
print("\n" + "="*85)
print(f"{'Ordo (n)':<10}{'Metode':<15}{'Waktu (s)':<15}{'Peak Mem (KB)':<18}{'Res. Error':<15}{'Iterasi'}")
print("="*85)
for i, n in enumerate(skala_tugas):
    print(f"{n:<10}{'Gauss':<15}{waktu_gauss[i]:<15.4f}{mem_gauss[i]:<18.2f}{err_gauss[i]:<15.2e}{'-':<15}")
    print(f"{'':<10}{'Gauss-Jordan':<15}{waktu_gj[i]:<15.4f}{mem_gj[i]:<18.2f}{err_gj[i]:<15.2e}{'-':<15}")
    print(f"{'':<10}{'Jacobi':<15}{waktu_jacobi[i]:<15.4f}{mem_jacobi[i]:<18.2f}{err_jacobi[i]:<15.2e}{it_jacobi[i]:<15}")
    print(f"{'':<10}{'Gauss-Seidel':<15}{waktu_gs[i]:<15.4f}{mem_gs[i]:<18.2f}{err_gs[i]:<15.2e}{it_gs[i]:<15}")
    print("-"*85)