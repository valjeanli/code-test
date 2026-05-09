#!/usr/bin/env python3
"""
Regime Detection: Non-parametric Online Market Regime Detection
Based on Horvath & Issa (2023), arXiv:2306.15835 / SSRN 3947905

"Non-parametric online market regime detection and regime clustering
for multidimensional and path-dependent data structures"

Core methodology:
- Path signatures as features on path space (Sec. 2.1, Eq. 5-6)
- MMD for two-sample testing on signature features (Sec. 2.4, Eq. 16)
- Auto-evaluator: compare non-overlapping ensembles (Sec. 3.2.2, Def. 3.9, Eq. 30)
- Bootstrap threshold for change point detection (Def. 3.5)
- Regime clustering via k-means on signature features (Sec. 5)

Paper equations referenced:
  Eq.(5)/(6): Path signature (iterated integrals)
  Eq.(16): Unbiased MMD estimator
  Eq.(24): Sub-path partitioning SP_h
  Eq.(25): Ensemble paths EP_h  
  Eq.(28): Score matrix Lambda(s)
  Eq.(30): Auto-evaluator A_L(s)
"""

import numpy as np
import csv
import os
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass


def extract_sig_features(path: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Truncated path signature features.
    The signature of a path maps to iterated integrals (Eq. 5-6).
    Level 1: mean(dX_i) -> d features
    Level 2: int int dX_i dX_j -> d*d features (area/iterated integral)
    """
    # Increment the time-augmented path
    dX = np.diff(path, axis=0)  # (T-1, d)
    if len(dX) < 2:
        return np.zeros(dX.shape[1] * (1 + dX.shape[1]))

    d = dX.shape[1]

    # Level 1
    l1 = dX.mean(axis=0)

    # Level 2: iterated integral \int_0^T (\int_0^t dX_i) dX_j
    cumul = np.zeros(d)
    l2 = np.zeros((d, d))
    for t in range(len(dX)):
        dx = dX[t]
        l2 += np.outer(cumul, dx)
        cumul += dx
    l2 /= len(dX)  # normalize by path length

    feats = np.concatenate([l1, l2.flatten()])

    if level >= 3 and d <= 3:
        # Level 3: only for small d to avoid explosion
        cumul1 = np.zeros(d)
        cumul2 = np.zeros((d, d))
        l3 = np.zeros(d * d * d)
        for t in range(len(dX)):
            dx = dX[t]
            cumul2 += np.outer(cumul1, dx)
            l3 += np.repeat(cumul2.flatten(), d) * np.tile(dx, d * d)
            cumul1 += dx
        l3 /= len(dX)
        feats = np.concatenate([feats, l3])

    return feats


def transform_subpath(data_slice: np.ndarray) -> np.ndarray:
    """
    Phi = phi_incr o phi_time o phi_norm applied to a sub-path.
    data_slice: (h1, d) raw multi-channel data
    Returns: time-augmented, increment-based path
    """
    # Normalize each channel
    mu = data_slice.mean(axis=0)
    sigma = data_slice.std(axis=0) + 1e-12
    norm = (data_slice - mu) / sigma

    # Increments
    inc = np.diff(norm, axis=0)

    # Normalize increments  
    inc_sigma = inc.std(axis=0) + 1e-12
    inc = (inc - np.mean(inc, axis=0)) / inc_sigma

    # Add time channel
    T_inc = len(inc)
    t_col = np.arange(T_inc)[:, None] / max(T_inc - 1, 1)

    return np.concatenate([t_col, inc], axis=1)


def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """k(x,y) = exp(-gamma * ||x-y||^2)"""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    sq = np.maximum(XX + YY.T - 2.0 * X @ Y.T, 0.0)
    return np.exp(-gamma * sq)


def auto_gamma(X: np.ndarray) -> float:
    """gamma = 1 / median(|x_i - x_j|^2)"""
    if len(X) < 3:
        return 1.0
    n = min(len(X), 150)
    idx = np.random.choice(len(X), n, replace=False)
    S = X[idx]
    # Efficient pairwise sq distances
    d2 = np.sum((S[:, None] - S[None]) ** 2, axis=2)
    off = d2[~np.eye(n, dtype=bool)]
    med = float(np.median(off))
    if med < 1e-10:
        return 1.0
    return 1.0 / med


def mmd_unbiased(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """Unbiased MMD estimator (Eq. 16)."""
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return 0.0

    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)

    t_xx = (Kxx.sum() - Kxx.trace()) / (n * (n - 1))
    t_yy = (Kyy.sum() - Kyy.trace()) / (m * (m - 1))
    t_xy = Kxy.sum() / (n * m)

    val = t_xx + t_yy - 2 * t_xy
    return max(val, 0.0)


def bootstrap_c_alpha(ref_features: np.ndarray, ens_size: int,
                      alpha: float = 0.05, n_boot: int = 500,
                      gamma: Optional[float] = None) -> float:
    """Compute critical value from bootstrap null distribution (Def. 3.5)."""
    if gamma is None:
        gamma = auto_gamma(ref_features)
        if gamma < 1e-6:
            gamma = 1.0

    N = len(ref_features)
    mmds = []
    for _ in range(n_boot):
        if N >= 2 * ens_size:
            idx = np.random.choice(N, 2 * ens_size, replace=False)
        else:
            idx = np.random.choice(N, 2 * ens_size, replace=True)
        A = ref_features[idx[:ens_size]]
        B = ref_features[idx[ens_size:]]
        mmds.append(mmd_unbiased(A, B, gamma))

    return float(np.percentile(mmds, (1 - alpha) * 100))


def regime_kmeans(features: np.ndarray, k: int = 3):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X = sc.fit_transform(features)
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X)

    info = {}
    for r in set(labels):
        mask = labels == r
        info[int(r)] = int(mask.sum())

    return labels, info


@dataclass
class Result:
    dates: List[str]
    labels: np.ndarray
    scores: np.ndarray
    cps: List[int]
    c_alpha: float
    gamma: float
    info: dict
    n_ens: int
    hp: dict


def detect(df, symbol: str,
           h1: int = 20, h2: int = 10,
           sig_level: int = 2, alpha: float = 0.05,
           n_bootstrap: int = 200) -> Optional[Result]:

    np.random.seed(42)

    sym = df[df['symbol'] == symbol].copy()
    sym = sym.sort_values('Date').reset_index(drop=True)

    min_needed = h1 * (2 * h2 + 5)
    if len(sym) < min_needed:
        return None

    prices = sym['Close'].values.astype(float)
    dates = sym['Date'].values
    T = len(prices)

    # Multi-channel data
    log_ret = np.zeros(T)
    log_ret[1:] = np.diff(np.log(np.abs(prices) + 1e-12))
    vols = sym['Volume'].values.astype(float)
    vol_ratio = vols / (vols.mean() + 1e-12)

    raw_data = np.column_stack([prices, log_ret, vol_ratio])  # (T, 3)

    # Step 1: Partition into sub-paths (Eq. 24)
    n_sub = T // h1
    subpaths = [raw_data[j * h1:(j + 1) * h1].copy() for j in range(n_sub)]

    # Step 2: Transform and extract signature for each sub-path
    sigs = []
    for sp in subpaths:
        t_path = transform_subpath(sp)
        sigs.append(extract_sig_features(t_path, level=sig_level))

    sigs = np.array(sigs)  # (n_sub, feat_dim)

    # Standardize
    mu_all = sigs.mean(axis=0)
    sigma_all = sigs.std(axis=0) + 1e-12
    sigs_norm = (sigs - mu_all) / sigma_all

    # Step 3: Build NON-OVERLAPPING ensembles (Eq. 25, but non-overlapping variant)
    # Each ensemble = h2 consecutive sub-path signatures, no overlap
    n_ens = n_sub // h2
    if n_ens < 3:
        return None

    ensemble_features = []  # list of (h2, feat_dim)
    ens_dates = []

    for k in range(n_ens):
        start = k * h2
        end = start + h2
        ens = sigs_norm[start:end]
        ensemble_features.append(ens)

        # Date midpoint
        raw_start = start * h1
        raw_mid = (raw_start + (end - 1) * h1) // 2
        ens_dates.append(str(dates[min(raw_mid, len(dates) - 1)])[:10])

    # Step 4: Auto-evaluator (Eq. 30)
    # Compare non-overlapping ensemble i to ensemble i-1 (lag=1)
    gamma = auto_gamma(ensemble_features[0])
    if gamma < 1e-6:
        gamma = 1.0

    scores = np.zeros(n_ens)
    for i in range(1, n_ens):
        scores[i] = mmd_unbiased(
            ensemble_features[i - 1],
            ensemble_features[i],
            gamma,
        )

    # Step 5: Bootstrap threshold (Def. 3.5)
    ref_concat = np.concatenate(ensemble_features[:n_ens // 2], axis=0)
    c_alpha = bootstrap_c_alpha(
        ref_concat,
        ens_size=min(h2, 10),
        alpha=alpha,
        n_boot=min(n_bootstrap, 500),
        gamma=gamma,
    )

    # Step 6: Change points
    cps = [i for i in range(1, n_ens) if scores[i] > c_alpha]

    # Step 7: Regime clustering (Sec. 5)
    avg_feats = np.array([ens.mean(axis=0) for ens in ensemble_features])
    labels, info = regime_kmeans(avg_feats, k=3)

    return Result(
        dates=ens_dates,
        labels=labels,
        scores=scores,
        cps=cps,
        c_alpha=c_alpha,
        gamma=gamma,
        info=info,
        n_ens=n_ens,
        hp={
            "h1": h1, "h2": h2, "sig_level": sig_level,
            "alpha": alpha, "gamma": gamma,
        },
    )


def fmt(r: Result, sym: str) -> str:
    L = [
        f"\n{'='*80}",
        f"REGIME DETECTION: {sym}",
        f"{'='*80}",
        f"Ensembles: {r.n_ens} | h1={r.hp['h1']} | h2={r.hp['h2']}",
        f"sig_level={r.hp['sig_level']} | gamma={r.gamma:.4f} | threshold={r.c_alpha:.6f}",
        f"Change points: {len(r.cps)}",
    ]
    L.append("\n--- Transitions ---")
    if r.cps:
        for cp in r.cps[:20]:
            if cp < len(r.dates):
                L.append(f"  {r.dates[cp]}: MMD={r.scores[cp]:.6f} (>{r.c_alpha:.6f})")
    else:
        valid = list(range(1, r.n_ens))
        top = sorted(valid, key=lambda i: r.scores[i], reverse=True)[:5]
        L.append("  None above threshold. Top 5:")
        for i in top:
            L.append(f"    {r.dates[i]}: MMD={r.scores[i]:.6f}")

    L.append("\n--- Regimes ---")
    for rr in sorted(set(r.labels)):
        mask = r.labels == rr
        n = int(mask.sum())
        pct = 100.0 * n / r.n_ens
        mm = r.scores[mask].mean()
        L.append(f"  Regime {rr}: {n}/{r.n_ens} ({pct:.0f}%), avg MMD={mm:.6f}")

    L.append("\n--- Timeline ---")
    step = max(1, r.n_ens // 60)
    L.append(f"  {'Date':<14} {'Reg':<4} {'MMD':>12} {'CP'}")
    L.append(f"  {'-'*14} {'-'*4} {'-'*12} {'--'}")
    for i in range(0, r.n_ens, step):
        cp = "*" if i in r.cps else ""
        L.append(f"  {r.dates[i]:<14} {r.labels[i]:<4} {r.scores[i]:>12.6f} {cp}")
    return "\n".join(L)


def main():
    print("=" * 80)
    print("Market Regime Detection — Horvath & Issa (2023), arXiv:2306.15835")
    print("=" * 80)

    paths = [
        '/opt/data/kanban/workspaces/t_dc995cda/sp500_data/sp500_ohlcv_5yr.csv',
        '/opt/data/kanban/workspaces/t_f9e0a695/sp500_data/sp500_ohlcv_5yr.csv',
    ]
    df = None
    for p in paths:
        if os.path.exists(p):
            print(f"Loading: {p}")
            df = pd.read_csv(p)
            break
    if df is None:
        print("ERROR: no data."); return

    print(f"Shape: {df.shape}, symbols: {df['symbol'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n")

    sc = df['symbol'].value_counts()
    symbols = sc.head(10).index.tolist()
    print(f"Analyzing: {symbols}\n")

    out = ["=" * 80,
           "REGIME DETECTION — arXiv 2306.15835 Implementation",
           "Horvath & Issa (2023)",
           "=" * 80]
    results = {}

    for sym in symbols:
        print(f"  {sym}...", end="", flush=True)
        r = detect(df, sym)
        if r:
            results[sym] = r
            print(" ok.")
            out.append(fmt(r, sym))
        else:
            print(" skip.")

    out.append("\n" + "=" * 80)
    out.append("CROSS-SYMBOL SUMMARY")
    out.append("=" * 80)
    out.append(f"Symbols: {len(results)}")
    out.append(f"Total change pts: {sum(len(r.cps) for r in results.values())}")
    out.append(f"Mean gamma: {np.mean([r.gamma for r in results.values()]):.4f}")
    out.append(f"Mean threshold: {np.mean([r.c_alpha for r in results.values()]):.6f}")
    out.append("\nChange points:")
    for sym, r in results.items():
        dts = [r.dates[c] for c in r.cps[:15] if c < len(r.dates)]
        out.append(f"  {sym}: {', '.join(dts) if dts else '(none detected)'}")

    text = "\n".join(out)
    print(text)

    outpath = '/opt/data/kanban/workspaces/t_ac96626f/paper-3947905/output_coder-qwen.txt'
    with open(outpath, 'w') as f:
        f.write(text)
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
