"""
solvers.py

Implementasi empat metode penyelesaian Sistem Persamaan Linier (SPL):
1. Eliminasi Gauss
2. Gauss-Jordan
3. Iterasi Jacobi
4. Iterasi Gauss-Seidel

Semua fungsi bersifat independen (tidak saling memanggil) dan mengikuti
pola input/output yang konsisten:

    Input  : A (matriks koefisien n x n), b (vektor konstanta n x 1)
    Output : (x, iter_count)
             - x          -> vektor solusi
             - iter_count -> jumlah iterasi (0 untuk metode langsung,
                              karena bukan proses berulang)

Sumber: Lampiran 1, Tugas Besar Metode Numerik
Kelompok 9 - Michael Alexander Newton, Arif Mufti Tharsa, Razqa Azaki
"""

from __future__ import annotations


def eliminasi_gauss_murni(
    A_orig: list[list[float]],
    b_orig: list[float],
) -> tuple[list[float], int]:
    """
    Menyelesaikan SPL Ax = b menggunakan metode Eliminasi Gauss.

    Algoritma bekerja dalam dua fase:
    1. Eliminasi maju (forward elimination): mengubah matriks A menjadi
       bentuk segitiga atas melalui Operasi Baris Elementer (OBE).
    2. Substitusi balik (back substitution): menghitung nilai x mulai
       dari baris terakhir ke baris pertama.

    Parameters
    ----------
    A_orig : list[list[float]]
        Matriks koefisien berukuran n x n.
    b_orig : list[float]
        Vektor konstanta berukuran n.

    Returns
    -------
    x : list[float]
        Vektor solusi hasil substitusi balik.
    iter_count : int
        Selalu 0, karena Eliminasi Gauss adalah metode langsung
        (bukan metode iteratif).

    Kompleksitas
    ------------
    O(n^3) untuk fase eliminasi, O(n^2) untuk substitusi balik.
    Total: O(n^3).
    """
    n = len(b_orig)
    # Deep copy manual agar data asli (A_orig, b_orig) tidak ikut berubah
    A = [[float(A_orig[i][j]) for j in range(n)] for i in range(n)]
    b = [float(b_orig[i]) for i in range(n)]

    # Fase 1: Eliminasi maju
    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]

    # Fase 2: Substitusi balik
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = b[i]
        for j in range(i + 1, n):
            total -= A[i][j] * x[j]
        x[i] = total / A[i][i]

    return x, 0


def eliminasi_gauss_jordan_murni(
    A_orig: list[list[float]],
    b_orig: list[float],
) -> tuple[list[float], int]:
    """
    Menyelesaikan SPL Ax = b menggunakan metode Gauss-Jordan.

    Berbeda dari Eliminasi Gauss, metode ini mereduksi matriks koefisien
    hingga menjadi matriks identitas (Reduced Row Echelon Form), sehingga
    solusi bisa langsung dibaca tanpa perlu substitusi balik.

    Parameters
    ----------
    A_orig : list[list[float]]
        Matriks koefisien berukuran n x n.
    b_orig : list[float]
        Vektor konstanta berukuran n.

    Returns
    -------
    x : list[float]
        Vektor solusi (nilai akhir dari b setelah direduksi).
    iter_count : int
        Selalu 0, karena Gauss-Jordan adalah metode langsung.

    Kompleksitas
    ------------
    O(n^3), dengan konstanta lebih besar dibanding Eliminasi Gauss karena
    melakukan eliminasi dua arah (atas dan bawah pivot).
    """
    n = len(b_orig)
    A = [[float(A_orig[i][j]) for j in range(n)] for i in range(n)]
    b = [float(b_orig[i]) for i in range(n)]

    for i in range(n):
        pivot = A[i][i]
        for k in range(i, n):
            A[i][k] /= pivot
        b[i] /= pivot

        for j in range(n):
            if i != j:
                factor = A[j][i]
                for k in range(i, n):
                    A[j][k] -= factor * A[i][k]
                b[j] -= factor * b[i]

    return b, 0


def iterasi_jacobi_murni(
    A: list[list[float]],
    b: list[float],
    max_iter: int = 15,
    tol: float = 1e-5,
) -> tuple[list[float], int]:
    """
    Menyelesaikan SPL Ax = b menggunakan metode iteratif Jacobi.

    Setiap variabel diperbarui menggunakan nilai dari iterasi
    sebelumnya (bukan nilai yang baru saja diperbarui dalam iterasi
    yang sama). Membutuhkan matriks A yang diagonal dominan ketat
    agar dijamin konvergen.

    Parameters
    ----------
    A : list[list[float]]
        Matriks koefisien berukuran n x n (idealnya diagonal dominan).
    b : list[float]
        Vektor konstanta berukuran n.
    max_iter : int, default=15
        Batas maksimum jumlah iterasi.
    tol : float, default=1e-5
        Toleransi konvergensi (selisih maksimum antar iterasi).

    Returns
    -------
    x : list[float]
        Vektor solusi hasil iterasi (aproksimasi).
    iter_count : int
        Jumlah iterasi yang benar-benar dijalankan hingga konvergen
        atau mencapai max_iter.

    Kompleksitas
    ------------
    O(n^2) per iterasi, sehingga total O(k * n^2) dengan k = iter_count.
    """
    n = len(b)
    x = [0.0] * n
    iter_count = 0

    for _ in range(max_iter):
        iter_count += 1
        x_new = [0.0] * n
        for i in range(n):
            total = 0.0
            for j in range(n):
                if i != j:
                    total += A[i][j] * x[j]
            x_new[i] = (b[i] - total) / A[i][i]

        diff = max(abs(x_new[k] - x[k]) for k in range(n))
        x = x_new
        if diff < tol:
            break

    return x, iter_count


def iterasi_gauss_seidel_murni(
    A: list[list[float]],
    b: list[float],
    max_iter: int = 15,
    tol: float = 1e-5,
) -> tuple[list[float], int]:
    """
    Menyelesaikan SPL Ax = b menggunakan metode iteratif Gauss-Seidel.

    Berbeda dari Jacobi, setiap variabel yang baru diperbarui langsung
    dipakai untuk menghitung variabel berikutnya dalam iterasi yang
    sama. Strategi ini umumnya mempercepat konvergensi dibanding Jacobi.

    Parameters
    ----------
    A : list[list[float]]
        Matriks koefisien berukuran n x n (idealnya diagonal dominan).
    b : list[float]
        Vektor konstanta berukuran n.
    max_iter : int, default=15
        Batas maksimum jumlah iterasi.
    tol : float, default=1e-5
        Toleransi konvergensi (selisih maksimum antar iterasi).

    Returns
    -------
    x : list[float]
        Vektor solusi hasil iterasi (aproksimasi).
    iter_count : int
        Jumlah iterasi yang benar-benar dijalankan hingga konvergen
        atau mencapai max_iter.

    Kompleksitas
    ------------
    O(n^2) per iterasi, sehingga total O(k * n^2) dengan k = iter_count.
    Umumnya k jauh lebih kecil dibanding Jacobi untuk toleransi yang sama.
    """
    n = len(b)
    x = [0.0] * n
    iter_count = 0

    for _ in range(max_iter):
        iter_count += 1
        x_old = x[:]
        for i in range(n):
            total = 0.0
            for j in range(n):
                if i != j:
                    total += A[i][j] * x[j]
            x[i] = (b[i] - total) / A[i][i]

        diff = max(abs(x[k] - x_old[k]) for k in range(n))
        if diff < tol:
            break

    return x, iter_count
