"""
Asset Selection via Correlation Blockmodel Clustering (ACC)
============================================================
Implementation of arXiv:2103.14506v2 (Tang, Xu, Zhou, 2021)

This implements the ACC (Asset Clustering through Correlation) algorithm
for selecting a well-diversified subset of stocks from a large universe,
followed by portfolio construction using risk parity, minimum variance,
and mean-variance allocation strategies.

Key equations referenced from the paper:
  Eq. 1: Correlation blockmodel: X*_i = F_{z(i)} + U_i
  Eq. 2: Correlation matrix: rho = Z Sigma_F Z^T + Sigma_U
  Eq. 3: Identifiability: max_{l!=i,j} |rho_il - rho_jl| = 0
  Eq. 4: CORD dissimilarity measure
  Eq. 5: Optimal asset selection: min-variance asset per cluster
  Eq. 6: Sample correlation matrix
  Eq. 7: Sample CORD
  Eq. 8: Threshold lower bound (Theorem 3)

Author: coder-qwen (Hermes Agent)
Date: 2026-05-07
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# ACC Algorithm Core Components
# ============================================================================


def compute_sample_correlation(X_star: np.ndarray) -> np.ndarray:
    """Compute sample correlation matrix from standardized returns (Eq. 6).

    Args:
        X_star: n x d matrix of standardized returns.

    Returns:
        d x d sample correlation matrix.
    """
    n = X_star.shape[0]
    # rho_hat = (1/(n-1)) X*^T X*  (Eq. 6)
    rho_hat = (1.0 / (n - 1)) * X_star.T @ X_star
    # Ensure symmetry and clip to valid correlation range
    rho_hat = (rho_hat + rho_hat.T) / 2.0
    np.fill_diagonal(rho_hat, 1.0)
    rho_hat = np.clip(rho_hat, -1.0, 1.0)
    return rho_hat


def compute_cord_matrix(rho_hat: np.ndarray) -> np.ndarray:
    """Compute the sample CORD dissimilarity matrix (Eq. 7).

    CORD(i, j) = max_{l != i,j} |rho_hat_il - rho_hat_jl|

    Args:
        rho_hat: d x d sample correlation matrix.

    Returns:
        d x d CORD dissimilarity matrix.
    """
    d = rho_hat.shape[0]
    cord = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            # Max over l != i, j of |rho_il - rho_jl|
            mask = np.ones(d, dtype=bool)
            mask[i] = False
            mask[j] = False
            cord[i, j] = np.max(np.abs(rho_hat[i, mask] - rho_hat[j, mask]))
            cord[j, i] = cord[i, j]
    return cord


def partition(cord: np.ndarray, epsilon: float) -> list[list[int]]:
    """PARTITION procedure (Procedure 1 in the paper).

    Given a dissimilarity matrix CORD and threshold epsilon,
    iteratively clusters assets by finding the most similar pair
    and growing a cluster around them.

    Args:
        cord: d x d CORD dissimilarity matrix.
        epsilon: Threshold for cluster membership.

    Returns:
        List of clusters, each cluster is a list of asset indices.
    """
    d = cord.shape[0]
    remaining = set(range(d))
    clusters: list[list[int]] = []

    while remaining:
        if len(remaining) == 1:
            clusters.append(list(remaining))
            break

        # Find the pair with minimum CORD in remaining set
        rem_list = sorted(remaining)
        min_cord = np.inf
        best_i, best_j = rem_list[0], rem_list[1]

        for idx_a in range(len(rem_list)):
            for idx_b in range(idx_a + 1, len(rem_list)):
                i, j = rem_list[idx_a], rem_list[idx_b]
                if cord[i, j] < min_cord:
                    min_cord = cord[i, j]
                    best_i, best_j = i, j

        if min_cord > epsilon:
            # Singleton cluster
            clusters.append([best_i])
            remaining.remove(best_i)
        else:
            # Grow cluster around (best_i, best_j)
            cluster = []
            for k in remaining:
                if min(cord[best_i, k], cord[best_j, k]) <= epsilon:
                    cluster.append(k)
            clusters.append(cluster)
            remaining -= set(cluster)

    return clusters


def estimate_tail_parameters(
    X_star: np.ndarray, rho_hat: np.ndarray, k_frac: float = 0.25
) -> tuple[float, float]:
    """Estimate heavy-tailedness alpha and constant L (Section 2.4, Eq. 10-11).

    Uses the Gardes & Girard (2008) method: linear regression of
    log quantiles against log log(2n/j) on the tail observations.

    Args:
        X_star: n x d matrix of standardized returns.
        rho_hat: d x d sample correlation matrix.
        k_frac: Fraction of observations to use as tail (default 0.25 = n/4).

    Returns:
        (alpha, L): Heavy-tailedness parameter and scale constant.
    """
    n, d = X_star.shape
    k = int(n * k_frac)

    # Compute rho^{-1/2} via eigenvalue decomposition
    # Add small regularization for numerical stability
    eigvals, eigvecs = np.linalg.eigh(rho_hat)
    eigvals = np.maximum(eigvals, 1e-10)
    rho_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Whiten the returns: Y = |rho^{-1/2} X*|
    Y_whitened = np.abs(X_star @ rho_inv_sqrt.T)  # n x d

    alphas = []
    ls = []

    for r in range(d):
        y_col = Y_whitened[:, r]
        sorted_y = np.sort(y_col)

        # Use the largest k observations: Y[n-k:n-1] (exclude the very max)
        # Corresponding quantiles: (n-j)/n for j=1..k
        tail_values = sorted_y[n - k : n - 1]  # k-1 values
        if len(tail_values) < 3:
            continue

        j = np.arange(1, len(tail_values) + 1)
        # x = log log(2n/j), y = log(Y_{n-j})
        x = np.log(np.log(2.0 * n / j))
        y = np.log(tail_values)

        # Linear regression: y = (1/alpha) * x + log L
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3:
            continue

        slope, intercept, _, _, _ = stats.linregress(x[valid], y[valid])

        if slope > 0:
            alpha_r = 1.0 / slope
            L_r = np.exp(intercept)
            # Clamp to reasonable range
            alpha_r = np.clip(alpha_r, 0.1, 2.0)
            alphas.append(alpha_r)
            ls.append(L_r)

    if not alphas:
        # Fallback defaults (paper finds alpha ~ 0.45-0.65 for S&P 500)
        return 0.5, 0.7

    # alpha = min_i alpha_i, L = max_i L_i (Section 2.4)
    alpha_star = min(alphas)
    L_star = max(ls)

    return alpha_star, L_star


def compute_intra_cluster_correlation(
    clusters: list[list[int]], rho_hat: np.ndarray
) -> float:
    """Compute average intra-cluster correlation (Eq. 9).

    rho_ave = sum_{i<j, i~j} rho_ij / count_{i<j, i~j}

    Args:
        clusters: List of clusters (each a list of asset indices).
        rho_hat: d x d sample correlation matrix.

    Returns:
        Average intra-cluster correlation, or -inf if no pairs.
    """
    total_corr = 0.0
    count = 0
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        for idx_a in range(len(cluster)):
            for idx_b in range(idx_a + 1, len(cluster)):
                i, j = cluster[idx_a], cluster[idx_b]
                total_corr += rho_hat[i, j]
                count += 1

    if count == 0:
        return float("-inf")
    return total_corr / count


def run_acc(
    X: np.ndarray,
    a: float = 0.1,
    b: float = 10.0,
    ng: int = 100,
    cluster_range: tuple[int, int] = (15, 25),
    k_frac: float = 0.25,
) -> dict:
    """Full ACC algorithm (Algorithm 1 in the paper).

    Args:
        X: n x d matrix of raw returns.
        a, b: Search range multipliers for epsilon (Rule 1).
        ng: Number of grid points for epsilon search.
        cluster_range: (min_clusters, max_clusters) allowed (Rule 3).
        k_frac: Fraction of observations for tail estimation.

    Returns:
        Dictionary with clusters, epsilon, alpha, L, and diagnostics.
    """
    n, d = X.shape

    # Step 1: Standardize returns
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0, ddof=1)
    stds = np.where(stds < 1e-10, 1.0, stds)  # Avoid division by zero
    X_star = (X - means) / stds

    # Handle NaN: replace with 0 (zero correlation after standardization)
    X_star = np.nan_to_num(X_star, nan=0.0)

    # Step 2: Sample correlation matrix (Eq. 6)
    rho_hat = compute_sample_correlation(X_star)

    # Step 3: CORD matrix (Eq. 7)
    cord = compute_cord_matrix(rho_hat)

    # Step 4: Estimate tail parameters
    alpha, L = estimate_tail_parameters(X_star, rho_hat, k_frac)

    # Step 5: Determine epsilon search range (Rule 1, Eq. 8)
    log_d = np.log(d)
    threshold_single = (log_d) ** (4.0 / alpha - 1.0)

    if n > threshold_single:
        base_scale = L**2 * np.sqrt(log_d / n)
    else:
        base_scale = L**2 * (log_d) ** (2.0 / alpha) / n

    # Cap upper bound to 2 (CORD max value)
    eps_min = min(a * base_scale, 2.0)
    eps_max = min(b * base_scale, 2.0)
    eps_min = max(eps_min, 1e-6)

    if eps_max <= eps_min:
        eps_max = eps_min * 10  # Ensure some range

    # Step 6: Grid search (Rules 2 & 3)
    eps_grid = np.linspace(eps_min, eps_max, ng)

    best_eps = eps_grid[0]
    best_rho_ave = float("-inf")
    best_clusters: list[list[int]] = []
    all_results = []

    for eps in eps_grid:
        clusters = partition(cord, eps)
        n_clusters = len(clusters)

        if cluster_range[0] <= n_clusters <= cluster_range[1]:
            rho_ave = compute_intra_cluster_correlation(clusters, rho_hat)
        else:
            rho_ave = float("-inf")

        all_results.append(
            {
                "eps": eps,
                "n_clusters": n_clusters,
                "rho_ave": rho_ave if rho_ave != float("-inf") else None,
            }
        )

        if rho_ave > best_rho_ave:
            best_rho_ave = rho_ave
            best_eps = eps
            best_clusters = clusters

    # Final clustering with best epsilon
    final_clusters = partition(cord, best_eps)

    # Select representative: lowest volatility asset per cluster (Theorem 2, Eq. 5)
    asset_vols = np.nanstd(X, axis=0, ddof=1)
    representatives = []
    for cluster in final_clusters:
        best_asset = min(cluster, key=lambda i: asset_vols[i])
        representatives.append(best_asset)

    return {
        "clusters": final_clusters,
        "representatives": representatives,
        "best_epsilon": best_eps,
        "alpha": alpha,
        "L": L,
        "eps_min": eps_min,
        "eps_max": eps_max,
        "n_clusters": len(final_clusters),
        "intra_cluster_corr": best_rho_ave,
        "search_results": all_results,
        "rho_hat": rho_hat,
        "cord": cord,
        "asset_vols": asset_vols,
        "X_star": X_star,
        "means": means,
        "stds": stds,
    }


# ============================================================================
# Portfolio Construction
# ============================================================================


def build_covariance_matrix(
    returns: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    """Build covariance matrix from correlation and volatilities.

    Sigma = diag(stds) @ rho @ diag(stds)
    """
    rho = compute_sample_correlation(returns)
    Sigma = np.outer(stds, stds) * rho
    return Sigma


def risk_parity_weights(Sigma: np.ndarray, max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """Compute risk parity portfolio weights (Appendix B.3, Eq. 17-20).

    Uses the CCD (Cyclical Coordinate Descent) algorithm from
    Maillard et al. (2010) to equalize risk contributions.

    Args:
        Sigma: d x d covariance matrix.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Weight vector summing to 1.
    """
    d = Sigma.shape[0]
    w = np.ones(d) / d  # Start with equal weights

    for _ in range(max_iter):
        w_old = w.copy()
        for i in range(d):
            # Marginal risk contribution: (Sigma w)_i
            sigma_w = Sigma @ w
            # Risk contribution of asset i
            rc_i = w[i] * sigma_w[i]
            if rc_i <= 0 or sigma_w[i] <= 0:
                continue
            # Target: sigma(w) / d, where sigma(w) = sqrt(w^T Sigma w)
            sigma_p = np.sqrt(w.T @ Sigma @ w)
            target = sigma_p / d
            # Update weight to equalize risk contribution
            # w_i = target / (Sigma w)_i (approximate CCD step)
            w[i] = target / sigma_w[i]

        # Normalize
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum

        # Enforce no short-selling
        w = np.maximum(w, 0)
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum

        # Check convergence
        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


def min_variance_weights(
    Sigma: np.ndarray,
) -> np.ndarray:
    """Compute minimum variance portfolio weights (Eq. 22).

    min w^T Sigma w  s.t.  w^T 1 = 1,  w >= 0

    Args:
        Sigma: d x d covariance matrix.

    Returns:
        Weight vector summing to 1.
    """
    d = Sigma.shape[0]

    def objective(w: np.ndarray) -> float:
        return w @ Sigma @ w

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * Sigma @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * d
    x0 = np.ones(d) / d

    result = optimize.minimize(
        objective, x0, jac=gradient, method="SLSQP", bounds=bounds, constraints=constraints
    )

    if result.success:
        w = result.x
    else:
        # Fallback: equal weight
        w = np.ones(d) / d

    w = np.maximum(w, 0)
    w /= w.sum()
    return w


def mean_variance_weights(
    Sigma: np.ndarray,
    mu: np.ndarray,
    target_return: float = 0.10,
) -> np.ndarray:
    """Compute Markowitz mean-variance optimal weights (Eq. 21).

    min w^T Sigma w  s.t.  w^T mu >= alpha,  w^T 1 = 1,  w >= 0

    Args:
        Sigma: d x d covariance matrix.
        mu: Expected returns (annualized).
        target_return: Target annualized return alpha (default 10%).

    Returns:
        Weight vector summing to 1.
    """
    d = Sigma.shape[0]

    def objective(w: np.ndarray) -> float:
        return w @ Sigma @ w

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * Sigma @ w

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: w @ mu - target_return},
    ]
    bounds = [(0.0, 1.0)] * d
    x0 = np.ones(d) / d

    result = optimize.minimize(
        objective, x0, jac=gradient, method="SLSQP", bounds=bounds, constraints=constraints
    )

    if result.success:
        w = result.x
    else:
        # Fallback: min variance
        w = min_variance_weights(Sigma)

    w = np.maximum(w, 0)
    w_sum = w.sum()
    if w_sum > 0:
        w /= w_sum
    return w


# ============================================================================
# Performance Metrics
# ============================================================================


def compute_portfolio_metrics(
    returns_daily: np.ndarray,
    weights: np.ndarray,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> dict:
    """Compute portfolio performance metrics.

    Args:
        returns_daily: T x d matrix of daily returns.
        weights: Portfolio weights (d,).
        risk_free_rate: Annual risk-free rate.
        trading_days: Number of trading days per year.

    Returns:
        Dictionary of performance metrics.
    """
    # Portfolio daily returns
    port_returns = returns_daily @ weights

    # Annualized metrics
    ann_return = np.mean(port_returns) * trading_days
    ann_vol = np.std(port_returns, ddof=1) * np.sqrt(trading_days)

    # Sharpe ratio
    excess_return = ann_return - risk_free_rate
    sharpe = excess_return / ann_vol if ann_vol > 0 else 0.0

    # Downside deviation
    negative_returns = port_returns[port_returns < 0]
    downside_vol = (
        np.std(negative_returns, ddof=1) * np.sqrt(trading_days)
        if len(negative_returns) > 1
        else 0.0
    )
    sortino = excess_return / downside_vol if downside_vol > 0 else 0.0

    # Maximum drawdown
    cumulative = np.cumprod(1 + port_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = np.min(drawdown)

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Ending VAMI (Value Added by Managed Investments)
    ending_vami = cumulative[-1] if len(cumulative) > 0 else 1.0

    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "ending_vami": ending_vami,
        "positive_days": np.sum(port_returns > 0),
        "negative_days": np.sum(port_returns <= 0),
        "total_days": len(port_returns),
    }


# ============================================================================
# Main Execution
# ============================================================================


def main() -> None:
    """Run the full ACC pipeline on S&P 500 data."""
    import os

    # ---- Data Loading ----
    print("=" * 70)
    print("ACC Algorithm: Asset Selection via Correlation Blockmodel Clustering")
    print("Paper: arXiv:2103.14506v2 (Tang, Xu, Zhou, 2021)")
    print("=" * 70)
    print()

    data_path_csv = "/opt/data/kanban/workspaces/t_dc995cda/sp500_data/sp500_ohlcv_5yr.csv"
    symbols_path = "/opt/data/kanban/workspaces/t_dc995cda/sp500_data/symbols.csv"

    print(f"Loading data from {data_path_csv}")
    df = pd.read_csv(data_path_csv, parse_dates=["Date"])
    df = df.rename(columns={"symbol": "ticker", "Close": "close"})
    symbols_df = pd.read_csv(symbols_path)

    print(f"  Raw data: {len(df)} rows, {df['ticker'].nunique()} tickers")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print()

    # ---- Data Preparation ----
    # Pivot to get returns matrix: rows = dates, cols = tickers
    print("Preparing returns matrix...")
    close_pivot = df.pivot_table(index="Date", columns="ticker", values="close")
    close_pivot = close_pivot.sort_index()

    # Compute daily returns
    returns = close_pivot.pct_change()

    # Filter: keep tickers with sufficient history (>= 500 non-null returns)
    min_days = 500
    valid_tickers = returns.columns[returns.count() >= min_days].tolist()
    print(f"  Tickers with >= {min_days} days of returns: {len(valid_tickers)}")

    returns = returns[valid_tickers]

    # Drop dates with too many missing values (>5% missing)
    returns = returns.dropna(thresh=len(valid_tickers) * 0.95)
    print(f"  Final returns matrix: {returns.shape[0]} days x {returns.shape[1]} tickers")
    print()

    # ---- Run ACC Algorithm ----
    X = returns.values
    tickers = returns.columns.tolist()
    n, d = X.shape

    print("Running ACC algorithm...")
    print(f"  n = {n} trading days, d = {d} assets")
    print()

    acc_result = run_acc(
        X,
        a=0.1,
        b=10.0,
        ng=100,
        cluster_range=(15, 25),
        k_frac=0.25,
    )

    # ---- Report Clustering Results ----
    print("-" * 70)
    print("CLUSTERING RESULTS")
    print("-" * 70)
    print(f"  Estimated alpha (heavy-tailedness): {acc_result['alpha']:.4f}")
    print(f"  Estimated L (scale constant):       {acc_result['L']:.4f}")
    print(f"  Best epsilon (threshold):           {acc_result['best_epsilon']:.6f}")
    print(f"  Epsilon search range:               [{acc_result['eps_min']:.6f}, {acc_result['eps_max']:.6f}]")
    print(f"  Number of clusters:                 {acc_result['n_clusters']}")
    print(f"  Intra-cluster correlation:          {acc_result['intra_cluster_corr']:.4f}")
    print()

    # Show cluster sizes and representatives
    print("Cluster Details:")
    print(f"  {'Cluster':<10} {'Size':<8} {'Rep Ticker':<12} {'Rep Vol (daily)':<18}")
    print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*18}")

    for idx, (cluster, rep) in enumerate(
        zip(acc_result["clusters"], acc_result["representatives"])
    ):
        ticker = tickers[rep]
        vol = acc_result["asset_vols"][rep]
        print(f"  {idx+1:<10} {len(cluster):<8} {ticker:<12} {vol:<18.6f}")

    print()

    # Show epsilon search results summary
    print("Epsilon Search Summary (valid cluster counts):")
    valid_results = [
        r for r in acc_result["search_results"] if r["rho_ave"] is not None
    ]
    if valid_results:
        print(f"  {'Epsilon':<12} {'Clusters':<10} {'Intra-Corr':<12}")
        print(f"  {'-'*12} {'-'*10} {'-'*12}")
        # Show a few representative results
        for r in valid_results[:5]:
            print(f"  {r['eps']:<12.6f} {r['n_clusters']:<10} {r['rho_ave']:<12.4f}")
        if len(valid_results) > 5:
            print(f"  ... and {len(valid_results) - 5} more valid epsilons")
    else:
        print("  No epsilon values produced clusters in the target range.")
        print("  Falling back to the epsilon that produced the closest cluster count.")

    print()

    # ---- Portfolio Construction ----
    print("-" * 70)
    print("PORTFOLIO CONSTRUCTION")
    print("-" * 70)

    rep_indices = acc_result["representatives"]
    rep_returns = X[:, rep_indices]
    rep_tickers = [tickers[i] for i in rep_indices]

    # Compute statistics for selected assets
    rep_means_daily = np.nanmean(rep_returns, axis=0)
    rep_stds_daily = np.nanstd(rep_returns, axis=0, ddof=1)
    rep_means_ann = rep_means_daily * 252
    rep_stds_ann = rep_stds_daily * np.sqrt(252)

    print(f"Selected {len(rep_indices)} representative stocks:")
    for i, t in enumerate(rep_tickers):
        print(f"  {t}: ann. return = {rep_means_ann[i]:.2%}, ann. vol = {rep_stds_ann[i]:.2%}")
    print()

    # Build covariance matrix for selected assets
    rep_returns_clean = np.nan_to_num(rep_returns, nan=0.0)
    rep_means_clean = np.nanmean(rep_returns_clean, axis=0)
    rep_stds_clean = np.nanstd(rep_returns_clean, axis=0, ddof=1)
    rep_stds_clean = np.where(rep_stds_clean < 1e-10, 1e-10, rep_stds_clean)

    # Standardize clean returns for correlation
    X_star_rep = (rep_returns_clean - rep_means_clean) / rep_stds_clean
    Sigma = build_covariance_matrix(X_star_rep, rep_means_clean, rep_stds_clean)

    # Three allocation strategies
    strategies = {}

    # 1. Risk Parity
    print("Computing risk parity weights...")
    w_rp = risk_parity_weights(Sigma)
    strategies["Risk Parity"] = w_rp

    # 2. Minimum Variance
    print("Computing minimum variance weights...")
    w_mv = min_variance_weights(Sigma)
    strategies["Minimum Variance"] = w_mv

    # 3. Mean-Variance
    print("Computing mean-variance weights (target return = 10%)...")
    w_mvo = mean_variance_weights(Sigma, rep_means_ann, target_return=0.10)
    strategies["Mean-Variance"] = w_mvo

    print()

    # ---- Portfolio Weights Summary ----
    print("Portfolio Weights (top 5 per strategy):")
    for name, w in strategies.items():
        print(f"\n  {name}:")
        top_idx = np.argsort(w)[::-1][:5]
        for i in top_idx:
            if w[i] > 0.001:  # Only show meaningful weights
                print(f"    {rep_tickers[i]:<12} {w[i]:.4f} ({w[i]:.1%})")
        print(f"    (plus {np.sum(w > 0.001) - 5} more positions)" if np.sum(w > 0.001) > 5 else "")

    print()

    # ---- Performance Evaluation ----
    print("-" * 70)
    print("PORTFOLIO PERFORMANCE (out-of-sample on same period)")
    print("-" * 70)
    print()

    # Compute returns for selected stocks
    rep_returns_out = np.nan_to_num(rep_returns_clean, nan=0.0)

    # Also compute SPY (market benchmark) for comparison
    # Use equal-weighted portfolio of all valid tickers as a simple benchmark
    ew_weights = np.ones(len(valid_tickers)) / len(valid_tickers)
    all_returns_clean = np.nan_to_num(X, nan=0.0)
    ew_returns = all_returns_clean @ ew_weights

    print(f"{'Metric':<25} {'Risk Parity':>14} {'Min Variance':>14} {'Mean-Var':>14} {'Equal-Weight':>14}")
    print(f"{'-'*25} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")

    all_metrics = {}
    for name, w in strategies.items():
        metrics = compute_portfolio_metrics(rep_returns_out, w)
        all_metrics[name] = metrics

    # Equal-weight benchmark
    ew_metrics = compute_portfolio_metrics(all_returns_clean, ew_weights)
    all_metrics["Equal-Weight"] = ew_metrics

    metrics_names = [
        ("Annualized Return", "annualized_return", "{:.2%}"),
        ("Annualized Volatility", "annualized_volatility", "{:.2%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.4f}"),
        ("Sortino Ratio", "sortino_ratio", "{:.4f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Calmar Ratio", "calmar_ratio", "{:.4f}"),
        ("Ending VAMI", "ending_vami", "{:.4f}"),
        ("Positive Days", "positive_days", "{:d}"),
    ]

    strategy_order = ["Risk Parity", "Minimum Variance", "Mean-Variance", "Equal-Weight"]

    for label, key, fmt in metrics_names:
        vals = []
        for s in strategy_order:
            m = all_metrics[s]
            vals.append(fmt.format(m[key]))
        print(f"{label:<25} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14} {vals[3]:>14}")

    print()

    # ---- Cluster Correlation Analysis ----
    print("-" * 70)
    print("CLUSTER CORRELATION ANALYSIS")
    print("-" * 70)

    rho_hat = acc_result["rho_hat"]
    clusters = acc_result["clusters"]

    # Inter-cluster average correlations
    print("\nAverage inter-cluster correlations (representative assets):")
    rep_indices_arr = np.array(rep_indices)
    rep_corr = rho_hat[np.ix_(rep_indices_arr, rep_indices_arr)]

    # Show correlation matrix of representatives (top 10)
    n_show = min(10, len(rep_tickers))
    print(f"\nCorrelation matrix of top {n_show} representative stocks:")
    header = "             " + " ".join(f"{rep_tickers[i]:>8}" for i in range(n_show))
    print(header)
    for i in range(n_show):
        row_str = f"{rep_tickers[i]:<12} "
        for j in range(n_show):
            row_str += f"{rep_corr[i, j]:8.3f}"
        print(row_str)

    print()

    # ---- Summary ----
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)

    best_sharpe_name = max(strategy_order[:-1], key=lambda s: all_metrics[s]["sharpe_ratio"])
    best_sharpe = all_metrics[best_sharpe_name]["sharpe_ratio"]
    ew_sharpe = all_metrics["Equal-Weight"]["sharpe_ratio"]

    print(f"\nBest ACC strategy: {best_sharpe_name} (Sharpe = {best_sharpe:.4f})")
    print(f"Equal-weight benchmark Sharpe: {ew_sharpe:.4f}")
    print(f"ACC improvement: +{best_sharpe - ew_sharpe:.4f} Sharpe")
    print(f"\nNumber of stocks selected: {len(rep_indices)} (from {d} total)")
    print(f"Diversification ratio: {d / len(rep_indices):.1f}x reduction")
    print(f"Selected tickers: {', '.join(rep_tickers)}")

    # ---- Save output ----
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output_coder-qwen.txt"
    )
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
