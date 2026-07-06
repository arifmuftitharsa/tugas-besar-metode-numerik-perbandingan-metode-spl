import numpy as np


# Menghitung residual error (selisih Ax - b) untuk mengukur akurasi solusi
def hitung_residual_error(A, x, b):
    n = len(b)
    residual = []
    for i in range(n):
        ax_i = 0.0
        for j in range(n):
            ax_i += float(A[i][j]) * float(x[j])
        residual.append(abs(ax_i - float(b[i])))
    return max(residual)

def hitung_r2(y_asli, y_pred):
    y_asli = np.array(y_asli, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    ss_res = np.sum((y_asli - y_pred) ** 2)
    ss_tot = np.sum((y_asli - np.mean(y_asli)) ** 2)
    return 1 - (ss_res / ss_tot)