# -*- coding: utf-8 -*-


import math
import time
import numpy as np
import pandas as pd

from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils.dataframe import dataframe_to_rows

# =========================================================
# Utils
# =========================================================
def clamp(lowb: np.ndarray, upb: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(X, lowb), upb)

def pairwise_sqdist_blas(X: np.ndarray) -> np.ndarray:
    """Pairwise squared distances using BLAS-friendly ops."""
    s = np.einsum("ij,ij->i", X, X)
    G = X @ X.T
    D2 = s[:, None] + s[None, :] - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    return D2

def scaled_radius(lowb: np.ndarray, upb: np.ndarray, rho: float) -> float:
    """Neighborhood radius scaled to the box diagonal."""
    diag = float(np.linalg.norm(upb - lowb))
    return max(1e-12, rho * diag)

def sr_from_neighbors(neigh: np.ndarray, r: np.ndarray, pbest_i: np.ndarray) -> np.ndarray:
    """
    Memory-safe accumulation of neighbor directions:
      sr_i = sum_{j in eta_i} (pbest_j - r_i) / ||pbest_j - r_i||
    neigh: (N,N) boolean, neigh[i,j]=True if j in eta_i
    """
    N, D = r.shape
    sr = np.zeros_like(r)
    # Loop over "source neighbor" j (adds its direction to all i that include j)
    for j in range(N):
        mask = neigh[:, j]
        if not np.any(mask):
            continue
        diff = pbest_i[j] - r[mask]             # (k,D)
        norm = np.linalg.norm(diff, axis=1)     # (k,)
        norm = np.where(norm > 0, norm, 1.0)
        sr[mask] += diff / norm[:, None]
    return sr

# =========================================================
# GSM and GSM-R (Rank-Based) — with scaled neighborhood radius and memory-safe sr
# =========================================================
def gsm_original_nd(r0, Z, lowb, upb, cost_fn, *, N, T, gamma=0.10, eps=0.10, Is=5, rho=0.03, return_hist=False):
    r = r0.copy()
    pbest_i = r.copy()
    pbest_f = np.full(N, np.inf)
    gbest = r[0].copy()
    gbest_f = np.inf
    R = scaled_radius(lowb, upb, rho)
    R2 = R * R
    hist = []

    for t in range(T):
        f = cost_fn(r).astype(float)
        f = np.nan_to_num(f, nan=np.inf, posinf=np.inf, neginf=np.inf)
        imp = f < pbest_f
        pbest_f[imp] = f[imp]
        pbest_i[imp] = r[imp]

        bi = int(np.argmin(f))
        if f[bi] < gbest_f:
            gbest_f = float(f[bi])
            gbest = r[bi].copy()

        mis = r[np.argsort(f)[:Is]]
        sigma = mis.std(axis=0, ddof=1 if Is > 1 else 0)

        D2 = pairwise_sqdist_blas(r)
        neigh = (D2 <= R2)
        np.fill_diagonal(neigh, False)

        deg = neigh.sum(axis=1).astype(float)
        sum_f_nb = neigh @ f
        mean_f_nb = np.zeros_like(f)
        nz = deg > 0
        mean_f_nb[nz] = sum_f_nb[nz] / deg[nz]

        x = (1 - eps) * f + eps * mean_f_nb
        x[~nz] = pbest_f[~nz]
        sx = neigh @ x
        sx = np.nan_to_num(sx, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        sr = sr_from_neighbors(neigh, r, pbest_i)

        beta = np.clip(x * sx, -1e3, 1e3)
        r = (gbest + Z[t] * sigma) + gamma * sr * beta[:, None]
        r = clamp(lowb, upb, r)

        if return_hist:
            hist.append(gbest_f)

    if return_hist:
        return float(gbest_f), np.array(hist, dtype=float)
    return float(gbest_f)


def gsm_rank_based_nd(r0, Z, lowb, upb, cost_fn, *, N, T,
                      gamma=0.12, eps=0.30, Is=5, rho=0.10, alpha=3.0, invert=True,
                      return_hist=False):
    r = r0.copy()
    pbest_i = r.copy()
    pbest_f = np.full(N, np.inf)
    gbest = r[0].copy()
    gbest_f = np.inf
    R = scaled_radius(lowb, upb, rho)
    R2 = R * R
    hist = []

    for t in range(T):
        f = cost_fn(r).astype(float)
        f = np.nan_to_num(f, nan=np.inf, posinf=np.inf, neginf=np.inf)
        imp = f < pbest_f
        pbest_f[imp] = f[imp]
        pbest_i[imp] = r[imp]

        bi = int(np.argmin(f))
        if f[bi] < gbest_f:
            gbest_f = float(f[bi])
            gbest = r[bi].copy()

        mis = r[np.argsort(f)[:Is]]
        sigma = mis.std(axis=0, ddof=1 if Is > 1 else 0)

        order = np.argsort(f)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(N)
        u = ranks / (N - 1)
        q = (1.0 - u) ** alpha if invert else (u ** alpha)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)

        D2 = pairwise_sqdist_blas(r)
        neigh = (D2 <= R2)
        np.fill_diagonal(neigh, False)

        deg = neigh.sum(axis=1).astype(float)
        sum_q_nb = neigh @ q
        mean_q_nb = np.zeros_like(q)
        nz = deg > 0
        mean_q_nb[nz] = sum_q_nb[nz] / deg[nz]

        x = (1 - eps) * q + eps * mean_q_nb
        x[~nz] = q[~nz]
        sx = neigh @ x
        sx = np.nan_to_num(sx, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        sr = sr_from_neighbors(neigh, r, pbest_i)

        beta = np.clip(x * sx, -1e3, 1e3)
        r = (gbest + Z[t] * sigma) + gamma * sr * beta[:, None]
        r = clamp(lowb, upb, r)

        if return_hist:
            hist.append(gbest_f)

    if return_hist:
        return float(gbest_f), np.array(hist, dtype=float)
    return float(gbest_f)


# =========================================================
# Baselines (DE/PSO/GA/ES)
# =========================================================
def de_rand_1_bin(cost_fn, lowb, upb, rng, *, N, T, F=0.7, CR=0.9, return_hist=False):
    D = len(lowb)
    X = lowb + (upb - lowb) * rng.random((N, D))
    f = cost_fn(X)
    hist = [float(np.min(f))]
    for _ in range(T):
        a = rng.integers(0, N, size=N)
        b = rng.integers(0, N, size=N)
        c = rng.integers(0, N, size=N)
        b = (b + (b == a)) % N
        c = (c + (c == a) + (c == b)) % N
        V = X[a] + F * (X[b] - X[c])
        V = clamp(lowb, upb, V)
        jrand = rng.integers(0, D, size=N)
        cross = rng.random((N, D)) < CR
        cross[np.arange(N), jrand] = True
        U = np.where(cross, V, X)
        fu = cost_fn(U)
        imp = fu < f
        X[imp] = U[imp]
        f[imp] = fu[imp]
        if return_hist:
            hist.append(float(np.min(f)))
    best = float(np.min(f))
    if return_hist:
        return best, np.array(hist, dtype=float)
    return best

def pso(cost_fn, lowb, upb, rng, *, N, T, w=0.7, c1=1.5, c2=1.5, return_hist=False):
    D = len(lowb)
    X = lowb + (upb - lowb) * rng.random((N, D))
    V = (rng.random((N, D)) - 0.5) * (upb - lowb)
    f = cost_fn(X)
    pbest = X.copy()
    pbest_f = f.copy()
    gbest = pbest[np.argmin(pbest_f)].copy()
    gbest_f = float(np.min(pbest_f))
    hist = [gbest_f]
    for _ in range(T):
        r1 = rng.random((N, D))
        r2 = rng.random((N, D))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
        X = clamp(lowb, upb, X + V)
        f = cost_fn(X)
        imp = f < pbest_f
        pbest[imp] = X[imp]
        pbest_f[imp] = f[imp]
        bi = int(np.argmin(pbest_f))
        if pbest_f[bi] < gbest_f:
            gbest_f = float(pbest_f[bi])
            gbest = pbest[bi].copy()
        if return_hist:
            hist.append(gbest_f)
    if return_hist:
        return float(gbest_f), np.array(hist, dtype=float)
    return float(gbest_f)

def ga_real_sbx(cost_fn, lowb, upb, rng, *, N, T, pc=0.9, pm=None, eta_c=15, eta_m=20, elitism=1, return_hist=False):
    D = len(lowb)
    if pm is None:
        pm = 1.0 / D
    X = lowb + (upb - lowb) * rng.random((N, D))
    f = cost_fn(X)
    hist = [float(np.min(f))]
    for _ in range(T):
        elite_idx = np.argsort(f)[:elitism]
        elite = X[elite_idx].copy()

        cand = rng.integers(0, N, size=(N, 2))
        best = cand[np.arange(N), np.argmin(f[cand], axis=1)]
        parents = X[best]
        off = parents.copy()

        for i in range(0, N - 1, 2):
            if rng.random() < pc:
                p1, p2 = parents[i], parents[i + 1]
                u = rng.random(D)
                beta = np.where(
                    u <= 0.5,
                    (2 * u) ** (1 / (eta_c + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta_c + 1)),
                )
                off[i] = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                off[i + 1] = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

        off = clamp(lowb, upb, off)

        mut_mask = rng.random((N, D)) < pm
        u = rng.random((N, D))
        delta = np.where(
            u < 0.5,
            (2 * u) ** (1 / (eta_m + 1)) - 1,
            1 - (2 * (1 - u)) ** (1 / (eta_m + 1)),
        )
        off = np.where(mut_mask, off + delta * (upb - lowb), off)
        off = clamp(lowb, upb, off)

        fo = cost_fn(off)
        X, f = off, fo

        if elitism > 0:
            worst = np.argsort(f)[-elitism:]
            X[worst] = elite
            f[worst] = cost_fn(elite)

        if return_hist:
            hist.append(float(np.min(f)))

    best = float(np.min(f))
    if return_hist:
        return best, np.array(hist, dtype=float)
    return best

def es_mu_lambda(cost_fn, lowb, upb, rng, *, mu=20, lam=60, T=20, sigma0=0.2, elitist=True, return_hist=False):
    D = len(lowb)
    tau = 1 / np.sqrt(2 * np.sqrt(D))
    X = lowb + (upb - lowb) * rng.random((mu, D))
    sig = np.full((mu, D), sigma0 * (upb - lowb))
    f = cost_fn(X)
    best = float(np.min(f))
    hist = [best]
    for _ in range(T):
        pidx = rng.integers(0, mu, size=lam)
        parents = X[pidx]
        psig = sig[pidx]
        sig_off = psig * np.exp(tau * rng.standard_normal((lam, D)))
        off = parents + sig_off * rng.standard_normal((lam, D))
        off = clamp(lowb, upb, off)
        fo = cost_fn(off)

        if elitist:
            allX = np.vstack([X, off])
            allf = np.concatenate([f, fo])
            alls = np.vstack([sig, sig_off])
            sel = np.argsort(allf)[:mu]
            X, f, sig = allX[sel], allf[sel], alls[sel]
        else:
            sel = np.argsort(fo)[:mu]
            X, f, sig = off[sel], fo[sel], sig_off[sel]

        best = min(best, float(np.min(f)))
        if return_hist:
            hist.append(best)

    if return_hist:
        return float(best), np.array(hist, dtype=float)
    return float(best)
