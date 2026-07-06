import numpy as np
import time
import matplotlib.pyplot as plt
import tracemalloc  # Indikator Memori bawaan Python

# ========================================================
# 1. 4 METODE NUMERIK MURNI + COUNTER ITERASI MANUAL
# ========================================================

def eliminasi_gauss_murni(A_orig, b_orig):
    n = len(b_orig)
    # Duplikasi data murni secara mendalam (deep copy manual)
    A = [[float(A_orig[i][j]) for j in range(n)] for i in range(n)]
    b = [float(b_orig[i]) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
            b[j] -= factor * b[i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = b[i]
        for j in range(i + 1, n):
            total -= A[i][j] * x[j]
        x[i] = total / A[i][i]
    return x, 0

def eliminasi_gauss_jordan_murni(A_orig, b_orig):
    n = len(b_orig)
    # Perbaikan krusial: Memastikan duplikasi tipe data float murni agar tidak merusak list asal
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

def iterasi_jacobi_murni(A, b, max_iter=15, tol=1e-5):
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

def iterasi_gauss_seidel_murni(A, b, max_iter=15, tol=1e-5):
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