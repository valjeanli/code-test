#!/usr/bin/env python3
"""
Asset Selection via Correlation Blockmodel Clustering
=====================================================
Implementation of the ACC (Asset Clustering through Correlation) algorithm
from Tang, Xu, and Zhou (2021) — arXiv:2103.14506.

This module implements the full ACC algorithm:
  1. Standardize returns
  2. Compute sample correlation matrix
  3. Compute CORD (Correlation Difference) dissimilarity matrix
  4. Estimate heavy-tailedness parameters α and L via tail regression
  5. Determine search range for threshold ε via Rule 1
  6. Grid search over ε, running PARTITION for each candidate
  7. Select ε maximizing intra-cluster correlation (Rule 2) subject to
     cluster count constraint (Rule 3)
  8. Return final partition

Reference equation numbers from the paper are noted in comments.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings

# ──────────────────────────────────────────────────────────────────────
# Constants from the paper (Section 3.2)
# ──────────────────────────────────────────────────────────────────────
DEFAULT_A = 0.1       # lower multiplier for ε search range (Rule 1)
DEFAULT_B = 10.0      # upper multiplier for ε search range (Rule 1)
DEFAULT_NG = 100      # number of grid points
DEFAULT_K_FRAC = 0.25 # k = n/4 tail observations for α estimation
DEFAULT_U = (15, 25)  # desired cluster count range (Rule 3)
EPS_CAP = 2.0         # cap on ε upper bound


@dataclass
class ACCResult:
    """Container for ACC algorithm results."""
    clusters: List[List[int]]       # each cluster = list of asset indices
    cluster_labels: np.ndarray      # label for each asset (0-indexed)
    n_clusters: int                 # number of clusters found
    epsilon: float                  # selected threshold ε
    alpha: float                    # estimated heavy-tailedness parameter
    L: float                        # estimated tail constant
    intra_corr: float               # average intra-cluster correlation (ρ̂_ave)
    cord_matrix: np.ndarray         # CORD dissimilarity matrix (d×d)
    corr_matrix: np.ndarray         # sample correlation matrix (d×d)
    search_range: Tuple[float, float]  # (ε_min, ε_max) used
    all_epsilons: np.ndarray        # all ε values tested
    all_intra_corrs: np.ndarray     # corresponding intra-cluster correlations
    all_n_clusters: np.ndarray      # corresponding cluster counts


# ══════════════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════


def prepare_returns(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    date_col: str = "Date",
    price_col: str = "Close",
    min_periods: int = 500,
    max_missing_pct: float = 0.05,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pivot raw OHLCV data into a returns matrix (dates × symbols).

    Follows the paper's data preparation (Section 3.1):
    - Pivot to (dates × symbols) matrix of closing prices
    - Drop symbols with < min_periods data points
    - Drop symbols with > max_missing_pct missing data
    - Compute daily returns: r_t = (P_t - P_{t-1}) / P_{t-1}
    - Forward-fill any remaining NaN (from non-trading days)

    Args:
        df: DataFrame with columns [date_col, symbol_col, price_col, ...]
        symbol_col: Column name for ticker/symbol
        date_col: Column name for date
        price_col: Column name for price (typically Close)
        min_periods: Minimum number of observations per symbol
        max_missing_pct: Maximum fraction of missing data allowed

    Returns:
        returns_df: (T × d) DataFrame of daily returns, dates as index
        symbols: List of symbol names (column order)
    """
    # Pivot to (dates × symbols)
    prices = df.pivot_table(
        index=date_col, columns=symbol_col, values=price_col, aggfunc="last"
    )
    prices = prices.sort_index()

    # Filter: minimum periods
    n_obs = prices.count()
    valid_symbols = n_obs[n_obs >= min_periods].index.tolist()
    prices = prices[valid_symbols]

    # Filter: maximum missing data percentage
    total_dates = len(prices)
    missing_pct = prices.isnull().sum() / total_dates
    valid_symbols = missing_pct[missing_pct <= max_missing_pct].index.tolist()
    prices = prices[valid_symbols]

    # Forward-fill remaining NaN (e.g., non-trading days within the range)
    prices = prices.ffill()

    # Compute simple daily returns: r_t = (P_t - P_{t-1}) / P_{t-1}
    returns = prices.pct_change().dropna()

    # Remove any rows that still have NaN
    returns = returns.dropna(axis=0, how="any")

    return returns, list(returns.columns)


# ══════════════════════════════════════════════════════════════════════
# 2. CORE ALGORITHM COMPONENTS
# ══════════════════════════════════════════════════════════════════════


def standardize_returns(returns: np.ndarray) -> np.ndarray:
    """
    Standardize returns column-wise: X* = (X - mean(X)) / std(X)

    Corresponds to the step in Algorithm 1 before computing ρ̂.

    Args:
        returns: (n × d) matrix of returns

    Returns:
        X_star: (n × d) matrix of standardized returns
    """
    mean = np.mean(returns, axis=0)
    std = np.std(returns, axis=0, ddof=0)  # population std for standardization
    # Avoid division by zero
    std[std < 1e-12] = 1.0
    return (returns - mean) / std


def compute_correlation_matrix(X_star: np.ndarray) -> np.ndarray:
    """
    Compute sample correlation matrix: ρ̂ = (1/(n-1)) (X*)^T X*  (Eq. 6)

    Args:
        X_star: (n × d) matrix of standardized returns

    Returns:
        corr: (d × d) correlation matrix
    """
    n = X_star.shape[0]
    corr = (X_star.T @ X_star) / (n - 1)
    # Ensure symmetry and unit diagonal (numerical precision)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    # Clip to valid correlation range
    corr = np.clip(corr, -1.0, 1.0)
    return corr


def compute_cord_matrix(corr: np.ndarray) -> np.ndarray:
    """
    Compute CORD (Correlation Difference) dissimilarity matrix.
    CORD(i,j) = max_{l ≠ i,j} |ρ_il - ρ_jl|  (Eq. 4/7)

    Uses an optimized approach: for each pair (i,j), take the max absolute
    difference between rows i and j of the correlation matrix, excluding
    columns i and j.

    Complexity: O(d³) in worst case, but uses vectorized operations.

    Args:
        corr: (d × d) correlation matrix

    Returns:
        cord: (d × d) symmetric matrix with CORD(i,j). Diagonal = 0.
    """
    d = corr.shape[0]
    cord = np.zeros((d, d))

    for i in range(d):
        for j in range(i + 1, d):
            # Absolute difference between row i and row j
            diff = np.abs(corr[i, :] - corr[j, :])
            # Exclude the diagonal elements i and j (self-correlations are always 1)
            diff[i] = 0.0
            diff[j] = 0.0
            cord_val = np.max(diff)
            cord[i, j] = cord_val
            cord[j, i] = cord_val

    return cord


def partition(dissimilarity: np.ndarray, epsilon: float) -> List[List[int]]:
    """
    PARTITION procedure (Procedure 1 in the paper).

    Groups assets into clusters based on the dissimilarity matrix D and
    threshold ε. Does NOT require the number of clusters K as input.

    Algorithm:
        S = {1, ..., d}   (unassigned set)
        while S not empty:
            if |S| == 1: singleton cluster
            else:
                (i*, j*) = argmin_{i,j∈S, i≠j} D(i,j)
                if D(i*, j*) > ε: singleton {i*}
                else: cluster = {k ∈ S: min(D(i*,k), D(j*,k)) ≤ ε}
    S = S - set(cluster)

    Args:
        dissimilarity: (d × d) symmetric dissimilarity matrix (CORD)
        epsilon: Threshold for merging

    Returns:
        List of clusters, each cluster is a list of asset indices
    """
    d = dissimilarity.shape[0]
    # Work with a copy to avoid modifying the input
    remaining = set(range(d))
    clusters = []

    while remaining:
        rem_list = list(remaining)

        if len(rem_list) == 1:
            # Singleton cluster (only one asset left)
            clusters.append([rem_list[0]])
            break

        # Find most similar pair among remaining assets
        # Only search among remaining assets
        best_val = np.inf
        i_star, j_star = -1, -1
        for idx_i, i in enumerate(rem_list):
            for j in rem_list[idx_i + 1 :]:
                val = dissimilarity[i, j]
                if val < best_val:
                    best_val = val
                    i_star, j_star = i, j

        if best_val > epsilon:
            # Core assets are too dissimilar → singleton
            clusters.append([i_star])
            remaining.discard(i_star)
        else:
            # Build cluster: all assets similar to EITHER core asset
            cluster = []
            for k in rem_list:
                if min(dissimilarity[i_star, k], dissimilarity[j_star, k]) <= epsilon:
                    cluster.append(k)
            clusters.append(cluster)
            remaining -= set(cluster)

    return clusters


def compute_intra_cluster_correlation(
    clusters: List[List[int]], corr: np.ndarray
) -> float:
    """
    Compute average intra-cluster correlation ρ̂_ave (Equation 9).

    ρ̂_ave = Σ_{i<j} 1[i~j] · ρ̂_ij  /  Σ_{i<j} 1[i~j]

    Args:
        clusters: List of clusters, each a list of asset indices
        corr: (d × d) correlation matrix

    Returns:
        Average intra-cluster correlation. Returns 1.0 if no intra-cluster pairs.
    """
    intra_pairs = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        c = sorted(cluster)
        for idx_i, i in enumerate(c):
            for j in c[idx_i + 1 :]:
                intra_pairs.append(corr[i, j])

    if not intra_pairs:
        return 1.0  # degenerate: all singletons, treat as perfect

    return float(np.mean(intra_pairs))


# ══════════════════════════════════════════════════════════════════════
# 3. HEAVY-TAILEDNESS ESTIMATION (Section 2.4)
# ══════════════════════════════════════════════════════════════════════


def estimate_alpha_L(
    X_star: np.ndarray, corr: np.ndarray, k: Optional[int] = None
) -> Tuple[float, float]:
    """
    Estimate heavy-tailedness parameter α and constant L via tail regression.

    Steps (Section 2.4):
        1. Compute Y = |ρ̂^(-1/2) X*| for each asset (coordinate-wise)
        2. For each asset r, sort Y_r and take k largest observations
        3. Regress log Y_r(n-j) against log log(2n/j) for j=1..k
        4. α_r = 1/slope, L_r = exp(intercept)
        5. α = min_r α_r, L = max_r L_r

    The quantile function for the boundary distribution is (Eq. 11):
        q(p) = L [log(2/(1-p))]^(1/α)
    Log-linear: log q(p) = (1/α) log log(2/(1-p)) + log L

    Args:
        X_star: (n × d) standardized returns
        corr: (d × d) sample correlation matrix ρ̂
        k: Number of tail observations (default: n/4, as in paper)

    Returns:
        alpha: Estimated heavy-tailedness parameter (min over all assets)
        L: Estimated constant (max over all assets)
    """
    n, d = X_star.shape

    if k is None:
        k = max(int(n * DEFAULT_K_FRAC), 10)
    k = min(k, n - 2)  # need at least 2 data points for regression

    # Step 1: Compute ρ̂^(-1/2) X*
    # Use eigendecomposition: ρ̂ = V Λ V^T → ρ̂^(-1/2) = V Λ^(-1/2) V^T
    eigenvalues, eigenvectors = np.linalg.eigh(corr)

    # Regularize: set very small eigenvalues to a minimum threshold
    eig_min = max(1e-10, eigenvalues[-1] * 1e-8)
    eigenvalues_clipped = np.maximum(eigenvalues, eig_min)

    # ρ̂^(-1/2) = V diag(λ^(-1/2)) V^T
    inv_sqrt_eigvals = 1.0 / np.sqrt(eigenvalues_clipped)
    corr_inv_sqrt = eigenvectors @ np.diag(inv_sqrt_eigvals) @ eigenvectors.T

    # Y = |ρ̂^(-1/2) X*| — element-wise absolute value
    Y = np.abs(X_star @ corr_inv_sqrt.T)  # (n × d) — Y[r] for asset r

    alpha_vals = []
    L_vals = []

    for r in range(d):
        Y_r = Y[:, r]
        Y_sorted = np.sort(Y_r)

        # Take k largest observations in DESCENDING order: Y(n-1), Y(n-2), ..., Y(n-k)
        # These pair with j=1..k where Y(n-j) approximates quantile q(1-j/n)
        # Y_sorted[-(k+1):-1] gives [Y(n-k), ..., Y(n-1)] ascending;
        # reverse to get [Y(n-1), Y(n-2), ..., Y(n-k)] for j=1..k
        tail_Y = Y_sorted[-(k + 1):-1][::-1]

        if len(tail_Y) < 3:
            alpha_vals.append(0.5)  # heavy-tailed default
            L_vals.append(1.0)
            continue

        # x = log log(2n/j) for j=1..k
        # j=1 → x = log log(2n), pairs with Y(n-1)
        # j=k → x = log log(2n/k), pairs with Y(n-k)
        actual_k = len(tail_Y)
        j_vals = np.arange(1, actual_k + 1)
        x = np.log(np.log(2.0 * n / j_vals))

        # y = log Y(n-j)
        y = np.log(np.maximum(tail_Y, 1e-15))

        # Linear regression: y = slope * x + intercept
        # Theory: log q = (1/α) log log(2/(1-p)) + log L
        # slope = 1/α, intercept = log L
        # Both x and y should decrease with j → slope > 0
        if len(x) >= 3 and np.std(x) > 1e-10:
            slope, intercept = np.polyfit(x, y, 1)
            # Enforce positive slope (theoretical constraint)
            if slope <= 0:
                slope = 1e-6
            alpha_r = 1.0 / slope
            # Clamp α to reasonable range (0 < α ≤ 2 per Assumption 1)
            alpha_r = np.clip(alpha_r, 0.01, 2.0)
            L_r = np.exp(intercept)
            # Clamp L to reasonable range
            L_r = np.clip(L_r, 0.1, 10.0)
        else:
            alpha_r = 0.5
            L_r = 1.0

        alpha_vals.append(alpha_r)
        L_vals.append(L_r)

    alpha = min(alpha_vals)
    L = max(L_vals)

    return alpha, L


# ══════════════════════════════════════════════════════════════════════
# 4. THRESHOLD TUNING (Rules 1-3, Section 2.3)
# ══════════════════════════════════════════════════════════════════════


def compute_search_range(
    n: int,
    d: int,
    alpha: float,
    L: float,
    cord: np.ndarray,
    a: float = DEFAULT_A,
    b: float = DEFAULT_B,
    cluster_range: Tuple[int, int] = DEFAULT_U,
) -> Tuple[float, float]:
    """
    Compute the search range for ε via Rule 1 (Section 2.3), refined with
    data-driven adjustments based on the CORD distribution.

    Rule 1 gives a theoretical range. In practice, the appropriate ε may be
    lower than the theoretical lower bound — we extend the search range
    downward to include CORD values that produce the desired cluster counts.

    Args:
        n: Number of observations
        d: Number of assets
        alpha: Heavy-tailedness parameter
        L: Tail constant
        cord: (d × d) CORD dissimilarity matrix
        a: Lower multiplier (default 0.1)
        b: Upper multiplier (default 10)
        cluster_range: (min_clusters, max_clusters)

    Returns:
        (ε_min, ε_max): Search range
    """
    log_d = np.log(d)

    # Rule 1: Theoretical bound
    if n > log_d ** (4.0 / alpha - 1.0):
        base = np.sqrt(log_d / n)
    else:
        base = (log_d ** (2.0 / alpha)) / n

    scale = L * L * base
    eps_min_theory = a * scale
    eps_max_theory = min(b * scale, EPS_CAP)

    # Data-driven adjustment: use CORD distribution
    # Get upper triangle CORD values
    d_dim = cord.shape[0]
    upper = cord[np.triu_indices(d_dim, k=1)]

    # Find approximate ε range that produces desired cluster counts
    # by scanning a few key percentiles
    pct_low = max(np.percentile(upper, 0.5), 1e-6)
    pct_high = np.percentile(upper, 99.5)

    # Use the minimum of theoretical and data-driven lower bound
    eps_min = min(eps_min_theory, pct_low)
    eps_min = max(eps_min, 1e-8)

    # Use the maximum of theoretical and data-driven upper bound
    eps_max = max(eps_max_theory, min(pct_high, EPS_CAP))
    eps_max = min(eps_max, EPS_CAP)
    eps_max = max(eps_max, eps_min * 1.01)  # ensure strictly greater

    return eps_min, eps_max


def grid_search_epsilon(
    cord: np.ndarray,
    corr: np.ndarray,
    eps_range: Tuple[float, float],
    ng: int = DEFAULT_NG,
    cluster_range: Tuple[int, int] = DEFAULT_U,
) -> Tuple[float, List[List[int]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Grid search over ε values to find the optimal threshold (Rules 2-3).

    For each ε in the range (ng evenly-spaced points):
        1. Run PARTITION(CORD, ε)
        2. If cluster count ∈ cluster_range: compute ρ̂_ave(ε)
        3. Else: ρ̂_ave(ε) = -∞

    Select ε that maximizes ρ̂_ave(ε).

    Args:
        cord: (d × d) CORD dissimilarity matrix
        corr: (d × d) correlation matrix
        eps_range: (ε_min, ε_max) from Rule 1
        ng: Number of grid points (default 100)
        cluster_range: (min_clusters, max_clusters) from Rule 3

    Returns:
        best_eps: Selected ε
        best_clusters: Cluster assignments from PARTITION(CORD, best_eps)
        all_eps: All ε values tested
        all_intra: Corresponding intra-cluster correlations
        all_nc: Corresponding cluster counts
    """
    eps_min, eps_max = eps_range
    epsilons = np.linspace(eps_min, eps_max, ng)

    best_eps = eps_min
    best_clusters = None
    best_intra = -np.inf

    all_intra = np.full(ng, -1.0)
    all_nc = np.zeros(ng, dtype=int)

    for idx, eps in enumerate(epsilons):
        clusters = partition(cord, eps)
        n_clusters = len(clusters)
        all_nc[idx] = n_clusters

        if cluster_range[0] <= n_clusters <= cluster_range[1]:
            intra = compute_intra_cluster_correlation(clusters, corr)
            all_intra[idx] = intra

            if intra > best_intra:
                best_intra = intra
                best_eps = eps
                best_clusters = clusters
        # else: all_intra[idx] stays at -1 (rejected)

    # Fallback: if no ε satisfies cluster range, pick the one closest to range
    if best_clusters is None:
        # Find ε whose cluster count is closest to the middle of the range
        target = (cluster_range[0] + cluster_range[1]) / 2
        closest_idx = np.argmin(np.abs(all_nc.astype(float) - target))
        best_eps = epsilons[closest_idx]
        best_clusters = partition(cord, best_eps)
        best_intra = compute_intra_cluster_correlation(best_clusters, corr)

    return best_eps, best_clusters, epsilons, all_intra, all_nc


# ══════════════════════════════════════════════════════════════════════
# 5. MAIN ACC ALGORITHM (Algorithm 1)
# ══════════════════════════════════════════════════════════════════════


def acc_algorithm(
    returns: np.ndarray,
    a: float = DEFAULT_A,
    b: float = DEFAULT_B,
    ng: int = DEFAULT_NG,
    cluster_range: Tuple[int, int] = DEFAULT_U,
    k: Optional[int] = None,
    random_seed: int = 42,
) -> ACCResult:
    """
    ACC (Asset Clustering through Correlation) Algorithm — Algorithm 1.

    Full pipeline:
        1. Standardize returns
        2. Compute sample correlation matrix ρ̂  (Eq. 6)
        3. Compute CORD dissimilarity matrix    (Eq. 7)
        4. Estimate α and L via tail regression (Section 2.4)
        5. Compute ε search range via Rule 1    (Section 2.3)
        6. Grid search ε, maximize intra-cluster correlation (Rules 2-3)
        7. Return final partition

    Args:
        returns: (n × d) matrix of daily returns
        a: Lower multiplier for ε search range
        b: Upper multiplier for ε search range
        ng: Number of grid points for ε search
        cluster_range: (min, max) number of clusters (Rule 3)
        k: Tail observations for α estimation (None = n/4)
        random_seed: Seed for reproducibility

    Returns:
        ACCResult with clusters, ε, α, L, and diagnostics
    """
    np.random.seed(random_seed)
    n, d = returns.shape

    # Step 1: Standardize returns
    X_star = standardize_returns(returns)

    # Step 2: Sample correlation matrix (Eq. 6)
    corr = compute_correlation_matrix(X_star)

    # Step 3: CORD dissimilarity matrix (Eq. 7)
    cord = compute_cord_matrix(corr)

    # Step 4: Estimate α and L (Section 2.4)
    alpha, L = estimate_alpha_L(X_star, corr, k=k)

    # Step 5: Search range for ε (Rule 1, refined with CORD distribution)
    eps_range = compute_search_range(n, d, alpha, L, cord, a=a, b=b, cluster_range=cluster_range)

    # Step 6-7: Grid search + optimal ε (Rules 2-3)
    best_eps, best_clusters, all_eps, all_intra, all_nc = grid_search_epsilon(
        cord, corr, eps_range, ng=ng, cluster_range=cluster_range
    )

    # Build cluster labels
    cluster_labels = np.full(d, -1, dtype=int)
    for label, cluster in enumerate(best_clusters):
        for idx in cluster:
            cluster_labels[idx] = label

    intra_corr = compute_intra_cluster_correlation(best_clusters, corr)

    return ACCResult(
        clusters=best_clusters,
        cluster_labels=cluster_labels,
        n_clusters=len(best_clusters),
        epsilon=best_eps,
        alpha=alpha,
        L=L,
        intra_corr=intra_corr,
        cord_matrix=cord,
        corr_matrix=corr,
        search_range=eps_range,
        all_epsilons=all_eps,
        all_intra_corrs=all_intra,
        all_n_clusters=all_nc,
    )


# ══════════════════════════════════════════════════════════════════════
# 6. PORTFOLIO CONSTRUCTION (Section 3.3 / Theorem 2)
# ══════════════════════════════════════════════════════════════════════


def select_lowest_volatility_stocks(
    clusters: List[List[int]], returns: np.ndarray
) -> List[int]:
    """
    Select the lowest-volatility stock from each cluster (Theorem 2).

    Volatility is measured as the sample std of daily returns in the
    lookback window, annualized.

    Theorem 2: Among all portfolios picking one asset per cluster, the
    minimum-variance portfolio consists of the lowest-variance asset in
    each cluster.

    Args:
        clusters: List of clusters (lists of asset indices)
        returns: (n × d) returns matrix

    Returns:
        Selected asset indices (one per cluster)
    """
    # Annualized volatility for each asset
    vol = np.std(returns, axis=0, ddof=0) * np.sqrt(252)

    selected = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        # Find asset with minimum volatility in this cluster
        cluster_vols = {idx: vol[idx] for idx in cluster}
        best_idx = min(cluster_vols, key=cluster_vols.get)
        selected.append(best_idx)

    return selected


def compute_minimum_variance_weights(
    returns: np.ndarray, allow_short: bool = False
) -> np.ndarray:
    """
    Compute minimum-variance portfolio weights (no short-selling by default).

    Solves: min_w w^T Σ w  subject to Σ w_i = 1, w_i ≥ 0

    Uses quadratic programming via the closed-form solution for the
    unconstrained case, or a simple numerical solver for constrained.

    Args:
        returns: (n × d) returns for the selected assets
        allow_short: If True, allow negative weights

    Returns:
        Weights array (d,)
    """
    d = returns.shape[1]
    cov = np.cov(returns, rowvar=False)

    if allow_short:
        # Closed-form: w = Σ^(-1) 1 / (1^T Σ^(-1) 1)
        try:
            cov_inv = np.linalg.inv(cov)
            ones = np.ones(d)
            w = cov_inv @ ones / (ones @ cov_inv @ ones)
        except np.linalg.LinAlgError:
            w = np.ones(d) / d
    else:
        # No-short-sale: use simple iterative quadratic programming
        # (gradient projection method)
        w = np.ones(d) / d  # equal weight initialization
        lr = 0.01
        for _ in range(10000):
            grad = 2 * cov @ w
            w_new = w - lr * grad
            # Project onto simplex (non-negative, sum-to-one)
            w_new = np.maximum(w_new, 0)
            w_new = w_new / w_new.sum()
            if np.max(np.abs(w_new - w)) < 1e-8:
                break
            w = w_new

    return w


def compute_risk_parity_weights(returns: np.ndarray) -> np.ndarray:
    """
    Compute risk parity (equal risk contribution) portfolio weights.

    Risk parity: each asset contributes equally to portfolio risk.
    Uses the simple inverse-volatility approximation.

    Args:
        returns: (n × d) returns for the selected assets

    Returns:
        Weights array (d,)
    """
    vol = np.std(returns, axis=0, ddof=0)
    vol = np.maximum(vol, 1e-10)  # avoid division by zero
    inv_vol = 1.0 / vol
    w = inv_vol / inv_vol.sum()
    return w


# ══════════════════════════════════════════════════════════════════════
# 7. MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import os
    from datetime import datetime

    print("=" * 70)
    print("ACC Algorithm — Asset Selection via Correlation Blockmodel Clustering")
    print("Tang, Xu, Zhou (2021) — arXiv:2103.14506")
    print("=" * 70)
    print()

    # ── Load Data ─────────────────────────────────────────────────────
    data_path = "sp500_data/sp500_ohlcv_5yr.parquet"
    if not os.path.exists(data_path):
        # Try parent workspace
        data_path = "/opt/data/kanban/workspaces/t_dc995cda/sp500_data/sp500_ohlcv_5yr.parquet"
        if not os.path.exists(data_path):
            print(f"ERROR: Data file not found. Tried: {data_path}")
            sys.exit(1)

    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"  Raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Symbols: {df['symbol'].nunique()}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print()

    # ── Prepare Returns ───────────────────────────────────────────────
    print("Preparing returns matrix...")
    returns_df, symbols = prepare_returns(
        df,
        symbol_col="symbol",
        date_col="Date",
        price_col="Close",
        min_periods=500,
        max_missing_pct=0.05,
    )
    print(f"  Returns matrix: {returns_df.shape[0]} days × {returns_df.shape[1]} assets")
    print(f"  Date range: {returns_df.index[0]} to {returns_df.index[-1]}")
    print()

    # ── Run ACC on most recent window ─────────────────────────────────
    # Use the last 500 trading days as the lookback window
    n_lookback = min(500, len(returns_df))
    recent_returns = returns_df.iloc[-n_lookback:].values
    recent_symbols = list(returns_df.columns)

    print(f"Running ACC algorithm on last {n_lookback} trading days...")
    print(f"  Window: {returns_df.index[-n_lookback]} to {returns_df.index[-1]}")
    print(f"  Assets: {len(recent_symbols)}")
    print()

    result = acc_algorithm(
        recent_returns,
        a=DEFAULT_A,
        b=DEFAULT_B,
        ng=DEFAULT_NG,
        cluster_range=DEFAULT_U,
        k=max(int(n_lookback * DEFAULT_K_FRAC), 10),
    )

    # ── Print Results ─────────────────────────────────────────────────
    print("─" * 70)
    print("ACC RESULTS")
    print("─" * 70)
    print(f"  Number of assets (d):      {len(recent_symbols)}")
    print(f"  Number of observations (n): {n_lookback}")
    print(f"  Estimated α:                {result.alpha:.4f}")
    print(f"  Estimated L:                {result.L:.4f}")
    print(f"  Search range for ε:         [{result.search_range[0]:.6f}, {result.search_range[1]:.6f}]")
    print(f"  Selected ε:                 {result.epsilon:.6f}")
    print(f"  Number of clusters (K):     {result.n_clusters}")
    print(f"  Intra-cluster corr (ρ̂_ave): {result.intra_corr:.4f}")
    print()

    # ── Cluster Details ───────────────────────────────────────────────
    print("─" * 70)
    print("CLUSTER DETAILS")
    print("─" * 70)

    # Compute cluster-level statistics
    cluster_stats = []
    for label, cluster in enumerate(result.clusters):
        cluster_symbols = [recent_symbols[i] for i in cluster]
        # Intra-cluster correlations
        if len(cluster) > 1:
            intra_corrs = []
            for i_idx, i in enumerate(cluster):
                for j in cluster[i_idx + 1 :]:
                    intra_corrs.append(result.corr_matrix[i, j])
            avg_intra = np.mean(intra_corrs)
            min_intra = np.min(intra_corrs)
            max_intra = np.max(intra_corrs)
        else:
            avg_intra = 1.0
            min_intra = 1.0
            max_intra = 1.0

        # Volatility of assets in cluster
        vols = np.std(recent_returns[:, cluster], axis=0, ddof=0) * np.sqrt(252)
        min_vol_idx = cluster[np.argmin(vols)]
        min_vol_symbol = recent_symbols[min_vol_idx]

        cluster_stats.append(
            {
                "label": label + 1,
                "size": len(cluster),
                "avg_intra_corr": avg_intra,
                "min_intra_corr": min_intra,
                "max_intra_corr": max_intra,
                "symbols": cluster_symbols,
                "lowest_vol": min_vol_symbol,
                "lowest_vol_val": np.min(vols),
            }
        )

    # Sort by size descending
    cluster_stats.sort(key=lambda x: x["size"], reverse=True)

    for cs in cluster_stats:
        print(f"  Cluster {cs['label']:2d}: {cs['size']:3d} assets | "
              f"avg intra-ρ = {cs['avg_intra_corr']:.4f} "
              f"[{cs['min_intra_corr']:.4f}, {cs['max_intra_corr']:.4f}] | "
              f"lowest vol: {cs['lowest_vol']} ({cs['lowest_vol_val']:.1%})")
        # Print first 10 symbols
        syms = cs["symbols"][:10]
        if len(cs["symbols"]) > 10:
            syms_str = ", ".join(syms) + f", ... (+{len(cs['symbols']) - 10} more)"
        else:
            syms_str = ", ".join(syms)
        print(f"         {syms_str}")

    print()

    # ── Selected Portfolio ────────────────────────────────────────────
    print("─" * 70)
    print("PORTFOLIO SELECTION (lowest volatility per cluster, Theorem 2)")
    print("─" * 70)

    selected_indices = select_lowest_volatility_stocks(
        result.clusters, recent_returns
    )
    selected_symbols = [recent_symbols[i] for i in selected_indices]
    selected_vols = np.std(recent_returns[:, selected_indices], axis=0, ddof=0) * np.sqrt(252)

    print(f"  Selected {len(selected_symbols)} stocks:")
    for sym, vol in zip(selected_symbols, selected_vols):
        print(f"    {sym:6s}  annualized vol = {vol:.1%}")

    # Compute portfolio weights (minimum variance)
    selected_rets = recent_returns[:, selected_indices]

    if selected_rets.shape[1] >= 2:
        mv_weights = compute_minimum_variance_weights(selected_rets)
        rp_weights = compute_risk_parity_weights(selected_rets)

        print()
        print("  Minimum-Variance Weights:")
        for sym, w in zip(selected_symbols, mv_weights):
            if w > 0.001:
                print(f"    {sym:6s}  {w:.4f} ({w:.1%})")

        print()
        print("  Risk Parity (Inverse-Vol) Weights:")
        for sym, w in zip(selected_symbols, rp_weights):
            if w > 0.001:
                print(f"    {sym:6s}  {w:.4f} ({w:.1%})")

        # Portfolio risk metrics
        mv_port_ret = selected_rets @ mv_weights
        mv_ann_ret = np.mean(mv_port_ret) * 252
        mv_ann_vol = np.std(mv_port_ret, ddof=0) * np.sqrt(252)
        mv_sharpe = mv_ann_ret / mv_ann_vol if mv_ann_vol > 0 else 0

        print()
        print(f"  Minimum-Variance Portfolio (annualized):")
        print(f"    Return:  {mv_ann_ret:.2%}")
        print(f"    Vol:     {mv_ann_vol:.2%}")
        print(f"    Sharpe:  {mv_sharpe:.2f}")
    else:
        print()
        print("  (Single asset selected — skipping multi-asset portfolio construction)")

    # ── Grid Search Dashboard ─────────────────────────────────────────
    print()
    print("─" * 70)
    print("GRID SEARCH SUMMARY")
    print("─" * 70)
    print(f"  ε range: [{result.search_range[0]:.6f}, {result.search_range[1]:.6f}]")
    print(f"  {DEFAULT_NG} grid points")
    print()
    print(f"  {'ε':>12s}  {'Clusters':>8s}  {'ρ̂_ave':>10s}  Status")
    print(f"  {'─'*12}  {'─'*8}  {'─'*10}  {'─'*20}")

    for eps, nc, intra in zip(
        result.all_epsilons[:: max(1, DEFAULT_NG // 15)],
        result.all_n_clusters[:: max(1, DEFAULT_NG // 15)],
        result.all_intra_corrs[:: max(1, DEFAULT_NG // 15)],
    ):
        in_range = DEFAULT_U[0] <= nc <= DEFAULT_U[1]
        status = "✓ SELECTED" if abs(eps - result.epsilon) < 1e-10 else (
            "valid" if in_range else "rejected (out of range)"
        )
        marker = "→" if abs(eps - result.epsilon) < 1e-10 else " "
        intra_str = f"{intra:.6f}" if intra >= 0 else "N/A"
        print(f"  {marker} {eps:10.6f}  {nc:8d}  {intra_str:>10s}  {status}")

    print()
    print("=" * 70)
    print("Done.")
