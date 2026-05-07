"""
Asset Selection via Correlation Blockmodel Clustering (ACC Algorithm)
=====================================================================
Implementation of Tang, Xu & Zhou (2021), arXiv:2103.14506.

This script:
  1. Loads S&P 500 daily OHLCV data from the parent task.
  2. Computes daily returns and selects a lookback window (n=500 days).
  3. Runs the ACC algorithm to cluster assets.
  4. Selects one representative asset per cluster (lowest variance).
  5. Reports cluster statistics and selected assets.

Run with:
    python code_coder-kimi.py

Output is printed to stdout and also saved to output_coder-kimi.txt.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "/opt/data/kanban/workspaces/t_dc995cda/sp500_data/sp500_ohlcv_5yr.csv"
SYMBOLS_PATH = "/opt/data/kanban/workspaces/t_dc995cda/sp500_data/symbols.csv"
OUTPUT_PATH = "/opt/data/kanban/workspaces/t_20cade67/paper-2103.14506/output_coder-kimi.txt"

SEED = 42
np.random.seed(SEED)

# ACC hyperparameters (from paper's empirical analysis, Section 3.2)
LOOKBACK_DAYS = 500          # n  — number of observations
CLUSTER_RANGE = (15, 25)     # U  — allowed number of clusters
SEARCH_A = 0.1               # a  — lower multiplier for epsilon range
SEARCH_B = 10.0              # b  — upper multiplier for epsilon range
N_GRIDS = 100                # ng — number of grid points for epsilon search
K_TAIL = 125                 # k  — number of tail observations for alpha estimation
EPS_CAP = 2.0                # cap on epsilon upper bound

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_returns(path: str, n_days: int) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load daily Close prices from CSV, compute daily returns,
    and return the last n_days of valid returns.

    Returns
    -------
    returns : DataFrame (n_days x d)
        Daily returns matrix with assets as columns.
    tickers : list of str
        Ticker symbols corresponding to columns.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    # Pivot to wide format: index=Date, columns=symbol, values=Close
    prices = df.pivot(index="Date", columns="symbol", values="Close")

    # Compute daily returns: R_t = (P_t - P_{t-1}) / P_{t-1}
    returns = prices.pct_change().iloc[1:]

    # Drop columns with any NaN in the lookback window
    if len(returns) > n_days:
        returns = returns.iloc[-n_days:]

    # Require at least 95% non-NaN within the window (paper allows 5% missing)
    valid_cols = returns.columns[returns.isna().mean() <= 0.05].tolist()
    returns = returns[valid_cols].copy()

    # Forward-fill then backward-fill any remaining missing values
    returns = returns.ffill().bfill()

    # Drop any column that still has NaN (shouldn't happen after ffill/bfill)
    returns = returns.dropna(axis=1)

    tickers = returns.columns.tolist()
    return returns, tickers


# ---------------------------------------------------------------------------
# Core algorithm: ACC
# ---------------------------------------------------------------------------

def standardize_returns(X: np.ndarray) -> np.ndarray:
    """
    Column-wise standardization: subtract mean, divide by std.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Raw return matrix.

    Returns
    -------
    X_star : ndarray, shape (n, d)
        Standardized returns.
    """
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, ddof=1, keepdims=True)
    # Protect against zero std (identical series)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return (X - mu) / sigma


def sample_correlation_matrix(X_star: np.ndarray) -> np.ndarray:
    """
    Compute sample correlation matrix: (1/(n-1)) * X_star^T X_star.

    Parameters
    ----------
    X_star : ndarray, shape (n, d)
        Standardized returns.

    Returns
    -------
    rho_hat : ndarray, shape (d, d)
        Sample correlation matrix.
    """
    n = X_star.shape[0]
    return (X_star.T @ X_star) / (n - 1)


def compute_cord(rho_hat: np.ndarray) -> np.ndarray:
    """
    Compute sample CORD (Correlation Difference) matrix.

    CORD(i, j) = max_{l != i, j} |rho_hat_{i,l} - rho_hat_{j,l}|

    Parameters
    ----------
    rho_hat : ndarray, shape (d, d)
        Sample correlation matrix.

    Returns
    -------
    cord : ndarray, shape (d, d)
        Symmetric dissimilarity matrix.
    """
    d = rho_hat.shape[0]
    cord = np.zeros((d, d))

    # For each fixed i, compute max_{l != i, j} |rho[i,l] - rho[j,l]| over j
    for i in range(d):
        diff = np.abs(rho_hat[i, :] - rho_hat)  # shape (d, d), diff[j, l] = |rho[i,l] - rho[j,l]|
        diff[:, i] = -np.inf          # exclude l = i for all j
        np.fill_diagonal(diff, -np.inf)  # exclude l = j for pair (i, j)
        cord[i, :] = np.max(diff, axis=1)

    # Ensure symmetry (numerical safety)
    cord = np.maximum(cord, cord.T)
    np.fill_diagonal(cord, 0.0)
    return cord


def estimate_heavy_tailedness(
    X_star: np.ndarray,
    rho_hat: np.ndarray,
    k: int,
) -> Tuple[float, float]:
    """
    Estimate alpha and L via linear regression on tail quantiles.

    For each asset r:
      Y_r = |(rho^{-1/2} X*)_r|  (length n)
      Sort Y_r ascending.
      Regress log(Y_{r,(n-j)}) on log(log(2n/j)) for j = 1, ..., k.
      slope s -> alpha_r = 1/s,  intercept a -> L_r = exp(a)

    Returns
    -------
    alpha : float
        min_r alpha_r  (most heavy-tailed)
    L : float
        max_r L_r
    """
    n, d = X_star.shape

    # Compute rho^{-1/2} via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(rho_hat)
    # Add small jitter if eigenvalues are non-positive (numerical safety)
    eigvals = np.maximum(eigvals, 1e-8)
    rho_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Whitened data: W = X_star @ rho_inv_sqrt  (n x d)
    W = X_star @ rho_inv_sqrt

    alphas = np.zeros(d)
    Ls = np.zeros(d)

    j = np.arange(1, k + 1, dtype=float)
    x = np.log(np.log(2.0 * n / j))  # shape (k,)

    for r in range(d):
        Y_r = np.abs(W[:, r])
        Y_sorted = np.sort(Y_r)  # ascending

        # Y_{r,(n-j)} for j = 1, ..., k  => indices n-2, n-3, ..., n-k-1 in 0-indexed
        y = np.log(Y_sorted[n - 1 - j.astype(int)])

        # Linear regression y = s * x + intercept
        # Using normal equations for speed
        X_design = np.vstack([x, np.ones_like(x)]).T  # (k, 2)
        coeffs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        s = coeffs[0]
        intercept = coeffs[1]

        if s > 1e-8:
            alphas[r] = 1.0 / s
        else:
            alphas[r] = 2.0  # default to sub-Gaussian if slope is flat
        Ls[r] = np.exp(intercept)

    alpha = float(np.min(alphas))
    L = float(np.max(Ls))

    # Clamp alpha to valid range (0, 2]
    alpha = max(0.01, min(2.0, alpha))
    # L should be positive
    L = max(L, 1e-6)

    return alpha, L


def partition(D: np.ndarray, eps: float) -> List[List[int]]:
    """
    PARTITION procedure (Procedure 1 in the paper).
    Optimized with NumPy for the minimum-pair search.

    Parameters
    ----------
    D : ndarray, shape (d, d)
        Dissimilarity matrix.
    eps : float
        Threshold.

    Returns
    -------
    clusters : list of list of int
        Partition of [d].
    """
    d = D.shape[0]
    S_mask = np.ones(d, dtype=bool)
    clusters: List[List[int]] = []

    while S_mask.sum() > 0:
        S_idx = np.where(S_mask)[0]
        if len(S_idx) == 1:
            clusters.append([int(S_idx[0])])
            S_mask[S_idx[0]] = False
            continue

        # Extract submatrix for active set and find minimum off-diagonal entry
        D_sub = D[np.ix_(S_idx, S_idx)]
        np.fill_diagonal(D_sub, np.inf)
        min_val = float(D_sub.min())
        min_pos = np.unravel_index(int(D_sub.argmin()), D_sub.shape)
        min_i = int(S_idx[min_pos[0]])
        min_j = int(S_idx[min_pos[1]])

        if min_val > eps:
            clusters.append([min_i])
            S_mask[min_i] = False
        else:
            candidates = np.minimum(D[min_i, :], D[min_j, :]) <= eps
            G_idx = np.where(S_mask & candidates)[0].tolist()
            clusters.append(G_idx)
            S_mask[G_idx] = False

    return clusters


def intra_cluster_correlation(clusters: List[List[int]], rho_hat: np.ndarray) -> float:
    """
    Compute average intra-cluster correlation (Equation 9).

    Parameters
    ----------
    clusters : list of list of int
    rho_hat : ndarray, shape (d, d)

    Returns
    -------
    rho_ave : float
    """
    numer = 0.0
    denom = 0
    for G in clusters:
        m = len(G)
        if m < 2:
            continue
        for idx_i in range(m):
            for idx_j in range(idx_i + 1, m):
                i, j = G[idx_i], G[idx_j]
                numer += rho_hat[i, j]
                denom += 1
    if denom == 0:
        return -np.inf
    return numer / denom


def acc_algorithm(
    X: np.ndarray,
    a: float = SEARCH_A,
    b: float = SEARCH_B,
    ng: int = N_GRIDS,
    cluster_range: Tuple[int, int] = CLUSTER_RANGE,
    k_tail: int = K_TAIL,
    eps_cap: float = EPS_CAP,
) -> Tuple[List[List[int]], dict]:
    """
    Asset Clustering through Correlation (ACC) — Algorithm 1.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Raw return matrix.
    a, b : float
        Multipliers for epsilon search range.
    ng : int
        Number of grid points.
    cluster_range : (int, int)
        Allowed number of clusters.
    k_tail : int
        Number of tail observations for alpha estimation.
    eps_cap : float
        Cap on epsilon upper bound.

    Returns
    -------
    best_clusters : list of list of int
        Selected partition.
    info : dict
        Diagnostics (alpha, L, best_eps, etc.).
    """
    n, d = X.shape
    k_min, k_max = cluster_range

    # Step 1: Standardize
    X_star = standardize_returns(X)

    # Step 2: Sample correlation
    rho_hat = sample_correlation_matrix(X_star)

    # Step 3: Sample CORD
    cord_hat = compute_cord(rho_hat)

    # Step 4: Estimate heavy-tailedness
    alpha, L = estimate_heavy_tailedness(X_star, rho_hat, k_tail)

    # Step 5: Determine search range for epsilon (Rule 1)
    logd = np.log(d)
    threshold = (logd) ** (4.0 / alpha - 1.0)

    if n > threshold:
        base = L ** 2 * np.sqrt(logd / n)
    else:
        base = L ** 2 * (logd ** (2.0 / alpha)) / n

    eps_low = a * base
    eps_high = min(b * base, eps_cap)

    # Ensure valid range
    if eps_low >= eps_high:
        eps_low = eps_high / 10.0

    # Step 6: Grid search over epsilon (Rules 2 & 3)
    epsilons = np.linspace(eps_low, eps_high, ng)

    best_eps = None
    best_rho_ave = -np.inf
    best_clusters = None
    best_k = None

    # Fallback tracking if no epsilon satisfies cluster_range
    fallback_candidates = []

    for eps in epsilons:
        clusters = partition(cord_hat, eps)
        k = len(clusters)

        if k_min <= k <= k_max:
            rho_ave = intra_cluster_correlation(clusters, rho_hat)
            if rho_ave > best_rho_ave:
                best_rho_ave = rho_ave
                best_eps = eps
                best_clusters = clusters
                best_k = k
        else:
            fallback_candidates.append((eps, clusters, k))

    # Fallback: if no epsilon in range, pick the one with K closest to midpoint of range
    if best_clusters is None:
        midpoint = (k_min + k_max) / 2.0
        fallback_candidates.sort(key=lambda t: abs(t[2] - midpoint))
        best_eps, best_clusters, best_k = fallback_candidates[0]
        best_rho_ave = intra_cluster_correlation(best_clusters, rho_hat)

    info = {
        "n": n,
        "d": d,
        "alpha": alpha,
        "L": L,
        "eps_low": eps_low,
        "eps_high": eps_high,
        "best_eps": best_eps,
        "best_k": best_k,
        "best_rho_ave": best_rho_ave,
    }
    return best_clusters, info


# ---------------------------------------------------------------------------
# Asset selection from clusters
# ---------------------------------------------------------------------------

def select_representatives(
    clusters: List[List[int]],
    X: np.ndarray,
    tickers: List[str],
) -> List[Tuple[str, int, float]]:
    """
    Select one representative asset per cluster.
    Following Theorem 2 (minimum-variance motivation), pick the asset
    with the lowest variance in each cluster.

    Returns
    -------
    selected : list of (ticker, cluster_id, variance)
    """
    selected = []
    for cid, G in enumerate(clusters):
        if not G:
            continue
        # Compute variances for assets in this cluster
        variances = np.var(X[:, G], axis=0, ddof=1)
        best_local_idx = int(np.argmin(variances))
        global_idx = G[best_local_idx]
        selected.append((tickers[global_idx], cid, float(variances[best_local_idx])))
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("ACC Algorithm — Tang, Xu & Zhou (2021)")
    print("arXiv:2103.14506")
    print("=" * 70)

    # Redirect stdout to both terminal and file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    out_file = open(OUTPUT_PATH, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, out_file)

    print("\n[1] Loading data...")
    returns_df, tickers = load_returns(DATA_PATH, LOOKBACK_DAYS)
    X = returns_df.values  # (n, d)
    n, d = X.shape
    print(f"    Lookback window: {n} days")
    print(f"    Assets after filtering: {d}")

    print("\n[2] Running ACC algorithm...")
    clusters, info = acc_algorithm(
        X,
        a=SEARCH_A,
        b=SEARCH_B,
        ng=N_GRIDS,
        cluster_range=CLUSTER_RANGE,
        k_tail=K_TAIL,
        eps_cap=EPS_CAP,
    )

    print(f"    Estimated alpha (heavy-tailedness): {info['alpha']:.4f}")
    print(f"    Estimated L: {info['L']:.4f}")
    print(f"    Epsilon search range: [{info['eps_low']:.6f}, {info['eps_high']:.6f}]")
    print(f"    Best epsilon: {info['best_eps']:.6f}")
    print(f"    Number of clusters found: {info['best_k']}")
    print(f"    Average intra-cluster correlation: {info['best_rho_ave']:.4f}")

    print("\n[3] Cluster sizes:")
    for cid, G in enumerate(clusters):
        print(f"    Cluster {cid + 1}: {len(G)} assets")

    print("\n[4] Selected representatives (lowest variance in each cluster):")
    selected = select_representatives(clusters, X, tickers)
    for ticker, cid, var in selected:
        print(f"    Cluster {cid + 1:2d}: {ticker:6s}  (variance = {var:.6f})")

    print("\n[5] Cluster compositions (top holdings):")
    for cid, G in enumerate(clusters):
        names = [tickers[i] for i in G]
        print(f"    Cluster {cid + 1:2d} ({len(G)} assets): {', '.join(names[:10])}" +
              (" ..." if len(names) > 10 else ""))

    # Compute some summary statistics
    print("\n[6] Summary statistics:")
    rho_hat = sample_correlation_matrix(standardize_returns(X))
    all_intra = []
    all_inter = []
    for cid1, G1 in enumerate(clusters):
        for cid2, G2 in enumerate(clusters):
            for i in G1:
                for j in G2:
                    if i >= j:
                        continue
                    if cid1 == cid2:
                        all_intra.append(rho_hat[i, j])
                    else:
                        all_inter.append(rho_hat[i, j])

    print(f"    Mean intra-cluster correlation: {np.mean(all_intra):.4f}")
    print(f"    Mean inter-cluster correlation: {np.mean(all_inter):.4f}")
    print(f"    Std intra-cluster correlation:  {np.std(all_intra):.4f}")
    print(f"    Std inter-cluster correlation:  {np.std(all_inter):.4f}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)

    sys.stdout = original_stdout
    out_file.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
