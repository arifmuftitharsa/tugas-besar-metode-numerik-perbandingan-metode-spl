import numpy as np
import matplotlib.pyplot as plt

# ========================================================
# 1. INDIKATOR DATA: INJEKSI DATA RIIL HARDWARE LAPTOP ANDA
# ========================================================
# Sumbu X: Ukuran matriks mulai dari 10, 100, lalu penambahan 100 hingga 2000
skala_tugas = np.array([10] + list(range(100, 2001, 100)))

# Indikator Waktu Eksekusi Riil Terminal Anda (Detik)
waktu_gauss = np.array([0.0002, 0.0738, 0.5413, 2.5428, 8.2640, 18.6725, 33.8389, 56.0864, 85.1632, 123.8920, 176.1273, 230.4503, 302.5916, 387.3668, 485.0216, 599.8173, 728.1785, 870.9082, 1040.8575, 1228.9003, 1432.3531])
waktu_gj    = np.array([0.0003, 0.1232, 0.9161, 4.7238, 15.1310, 33.0321, 60.0120, 98.4958, 150.3238, 215.6967, 299.1826, 401.7991, 524.4293, 671.0852, 834.4901, 1073.9752, 1254.0219, 1501.2565, 1794.9649, 2117.6431, 2466.1635])
waktu_jacobi= np.array([0.0003, 0.0250, 0.0951, 0.3002, 0.6416, 1.1669, 1.7542, 2.5020, 3.4000, 4.4010, 5.5638, 6.8570, 8.2943, 9.8490, 11.5933, 13.3422, 15.2795, 17.4281, 19.6598, 22.0871, 24.5053])
waktu_gs    = np.array([0.0002, 0.0101, 0.0375, 0.1170, 0.2472, 0.3740, 0.5783, 0.8555, 1.1000, 1.4665, 1.8368, 2.2538, 2.7356, 3.2300, 3.0064, 3.4959, 3.9769, 4.5471, 5.1017, 5.7204, 6.3845])

# Indikator Resource: Konsumsi Memori RAM Nyata Terminal Anda (KB)
mem_gauss   = np.array([2661.57, 3372.92, 5506.42, 9074.71, 14011.46, 20452.19, 28480.80, 37646.07, 48273.46, 60392.75, 74061.68, 88078.78, 104697.18, 122954.52, 140959.60, 162325.61, 185524.76, 207580.68, 234093.80, 258826.88, 284808.27])
mem_gj      = np.array([2661.77, 3375.03, 5508.05, 9075.89, 14012.02, 20451.85, 28479.77, 37644.33, 48271.38, 60389.87, 74058.18, 88074.67, 104692.61, 122949.49, 140953.85, 162318.48, 185516.32, 207571.00, 234083.34, 258815.64, 284796.28])
mem_jacobi  = np.array([2660.96, 3063.98, 4259.63, 6242.75, 9007.07, 12552.88, 16880.05, 21987.85, 27877.38, 34548.33, 42000.32, 50232.45, 59247.05, 69042.92, 79618.86, 90977.40, 103117.74, 116037.82, 129740.57, 144222.65, 159486.07])
mem_gs      = np.array([2661.16, 3063.37, 4263.96, 6247.80, 9013.03, 12559.49, 16887.44, 21996.03, 27886.58, 34558.12, 42010.78, 50243.69, 59259.07, 69055.73, 79632.45, 90991.77, 103133.14, 116053.81, 129757.34, 144240.14, 159504.35])

# Fungsi internal menghitung nilai R-Square (R²) secara manual
def hitung_r2(y_asli, y_pred):
    ss_res = np.sum((y_asli - y_pred) ** 2)
    ss_tot = np.sum((y_asli - np.mean(y_asli)) ** 2)
    return 1 - (ss_res / ss_tot)

# ========================================================
# 2. PROSES HITUNG RERATA & FITTING REGRESI POLINOMIAL
# ========================================================
metode_names = ['Gauss', 'Gauss-Jordan', 'Jacobi', 'Gauss-Seidel']
avg_waktu = [np.mean(waktu_gauss), np.mean(waktu_gj), np.mean(waktu_jacobi), np.mean(waktu_gs)]
avg_memori = [np.mean(mem_gauss), np.mean(mem_gj), np.mean(mem_jacobi), np.mean(mem_gs)]

# Menghitung garis tren interpolasi regresi derajat 4 untuk Grafik 1
pred4_gauss  = np.polyval(np.polyfit(skala_tugas, waktu_gauss, 4), skala_tugas)
pred4_gj     = np.polyval(np.polyfit(skala_tugas, waktu_gj, 4), skala_tugas)
pred4_jacobi = np.polyval(np.polyfit(skala_tugas, waktu_jacobi, 4), skala_tugas)
pred4_gs     = np.polyval(np.polyfit(skala_tugas, waktu_gs, 4), skala_tugas)

# ========================================================
# 3. KODE GRAFIK: PENYUSUNAN MULTI-PANEL GRAFIK (REVISI LAYOUT)
# ========================================================
plt.figure(figsize=(18, 6.2)) # Memperluas dimensi kanvas grafik
colors = ['#E53935', '#FB8C00', '#1E88E5', '#43A047'] # Paduan warna estetik

# --- [GRAFIK 1]: Ukuran Matriks vs Waktu Eksekusi (Kurva Kontinu) ---
plt.subplot(1, 3, 1)
plt.scatter(skala_tugas, waktu_gauss, color='#E53935', alpha=0.5, s=25)
plt.plot(skala_tugas, pred4_gauss, color='#E53935', linewidth=2, label=f'Gauss (R²={hitung_r2(waktu_gauss, pred4_gauss):.4f})')

plt.scatter(skala_tugas, waktu_gj, color='#FB8C00', alpha=0.5, s=25)
plt.plot(skala_tugas, pred4_gj, color='#FB8C00', linewidth=2, linestyle='--', label=f'Gauss-Jordan (R²={hitung_r2(waktu_gj, pred4_gj):.4f})')

plt.scatter(skala_tugas, waktu_jacobi, color='#1E88E5', alpha=0.5, s=25)
plt.plot(skala_tugas, pred4_jacobi, color='#1E88E5', linewidth=2, label=f'Jacobi (R²={hitung_r2(waktu_jacobi, pred4_jacobi):.4f})')

plt.scatter(skala_tugas, waktu_gs, color='#43A047', alpha=0.5, s=25)
plt.plot(skala_tugas, pred4_gs, color='#43A047', linewidth=2, linestyle='--', label=f'Gauss-Seidel (R²={hitung_r2(waktu_gs, pred4_gs):.4f})')

plt.title('Grafik 1: Ukuran Matriks vs Waktu Eksekusi', fontsize=11, fontweight='bold', pad=12)
plt.xlabel('Ukuran Matriks ($n$)')
plt.ylabel('Waktu Pemrosesan (Detik)')
plt.xlim(0, 2050)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper left', fontsize=9)

# --- [GRAFIK 2]: REVISI RUANG Metode vs Rata-rata Waktu Eksekusi (Bar Chart) ---
plt.subplot(1, 3, 2)
bars2 = plt.bar(metode_names, avg_waktu, color=colors, alpha=0.85, edgecolor='black', width=0.45)
plt.title('Grafik 2: Metode vs Rata-rata Waktu', fontsize=11, fontweight='bold', pad=20) # Ditambah jarak judul
plt.ylabel('Rata-rata Waktu (Detik)')
plt.grid(axis='y', linestyle=':', alpha=0.6)

# Melebarkan atap langit-langit sumbu-Y agar teks angka tidak terpotong judul
plt.ylim(0, 750)

for bar in bars2:
    yval = bar.get_height()
    # Menggeser letak teks vertikal ke posisi yval + 12 agar aman dari garis pembatas
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 12, f'{yval:.2f}s', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# --- [GRAFIK 3]: REVISI RUANG Metode vs Penggunaan Memori (Bar Chart) ---
plt.subplot(1, 3, 3)
bars3 = plt.bar(metode_names, avg_memori, color=colors, alpha=0.85, edgecolor='black', width=0.45)
plt.title('Grafik 3: Metode vs Penggunaan Memori', fontsize=11, fontweight='bold', pad=20) # Ditambah jarak judul
plt.ylabel('Rata-rata Peak Memori (KB)')
plt.grid(axis='y', linestyle=':', alpha=0.6)

# Melebarkan atap langit-langit sumbu-Y kapasitas memori agar lega
plt.ylim(0, 120000)

for bar in bars3:
    yval = bar.get_height()
    # Menggunakan yval + 2500 untuk menjaga jarak aman teks angka KB
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2500, f'{int(yval):,}\nKB', ha='center', va='bottom', fontsize=9, fontweight='bold')

# ========================================================
# LAYOUT OPTIMIZATION & SHOW IMAGE
# ========================================================
plt.suptitle('Analisis Komparatif Performa dan Resource 4 Metode Numerik Matriks ($n=10$ sampai $n=2000$)', fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()

# Opsi: hapus tanda pagar (#) pada baris di bawah jika ingin langsung otomatis menyimpan gambar ke laptop Anda
# plt.savefig('grafik_laporan_final.png', dpi=300, bbox_inches='tight')

plt.show()