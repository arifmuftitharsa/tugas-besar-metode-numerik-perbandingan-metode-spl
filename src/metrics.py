"""
metrics.py

Fungsi-fungsi metrik evaluasi untuk membandingkan hasil solver SPL:
1. Residual error (||Ax - b||_infinity)
2. Koefisien determinasi (R-squared)

Modul ini independen dan tidak mengimpor apa pun dari solvers.py.

Sumber: Lampiran 1, Tugas Besar Metode Numerik
Kelompok 9 - Michael Alexander Newton, Arif Mufti Tharsa, Razqa Azaki
"""

from __future__ import annotations

import numpy as np


def hitung_residual_error(
    A: list[list[float]],
    x: list[float],
    b: list[float],
) -> float:
    """
    Menghitung residual error maksimum ||Ax - b||_infinity.

    Untuk setiap baris i, dihitung selisih absolut antara hasil kali
    baris A dengan vektor x terhadap nilai b[i]. Nilai yang dikembalikan
    adalah nilai maksimum dari seluruh baris (norma tak hingga).

    Parameters
    ----------
    A : list[list[float]]
        Matriks koefisien berukuran n x n.
    x : list[float]
        Vektor solusi yang akan dievaluasi.
    b : list[float]
        Vektor konstanta berukuran n.

    Returns
    -------
    float
        Residual error maksimum (||Ax - b||_infinity).
    """
    n = len(b)
    max_error = 0.0
    for i in range(n):
        total = sum(A[i][j] * x[j] for j in range(n))
        error = abs(total - b[i])
        if error > max_error:
            max_error = error
    return max_error


def hitung_r2(
    y_asli: list[float],
    y_pred: list[float],
) -> float:
    """
    Menghitung koefisien determinasi (R-squared) antara nilai asli dan prediksi.

    Formula: R^2 = 1 - (SS_res / SS_tot)
    - SS_res = sum((y_asli - y_pred)^2)
    - SS_tot = sum((y_asli - mean(y_asli))^2)

    Parameters
    ----------
    y_asli : list[float]
        Nilai aktual/observasi.
    y_pred : list[float]
        Nilai hasil prediksi/estimasi.

    Returns
    -------
    float
        Nilai R-squared.
    """
    y_asli_arr = np.array(y_asli, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    ss_res = np.sum((y_asli_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_asli_arr - np.mean(y_asli_arr)) ** 2)

    return 1 - (ss_res / ss_tot)
