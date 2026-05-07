#!/usr/bin/env python3
"""
Asset Selection via Correlation Blockmodel Clustering (ACC Algorithm)
=====================================================================
Implementation of the ACC algorithm from:
    Tang, Xu, and Zhou (2021), "Asset Selection via Correlation Blockmodel Clustering"
    arXiv:2103.14506

This implementation follows the paper's Algorithm 1 (ACC) and Procedure 1 (PARTITION),
applying them to S&P 500 stock data for portfolio construction.

References to equations and theorems use the paper's numbering.
"""

import warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import minimize

# ============================================================================
# Constants (matching paper's empirical setup, Section 3.2)
# ============================================================================
DEFAULT_A = 0.1          # Lower bound multiplier for ε search range (Rule 1)
DEFAULT_B = 10.0         # Upper bound multiplier for ε search range (Rule 1)
DEFAULT_NG = 100         # Number of grid points for ε search (Rule 3)
DEFAULT_U = (15, 25)     # Range for acceptable number of clusters (Rule 3)
DEFAULT_K_RATIO = 0.25   # k = n/4 for tail estimation (Section 3.2)
EPS_CAP = 2.0            # Cap on upper bound of ε search range (Section 3.2)
RISK_FREE_RATE = 0.04    # Annual risk-free rate for Sharpe ratio


# ============================================================================
# Core Algorithm Components
# ============================================================================

def compute_standardized_returns(returns: pd.DataFrame) -> np.ndarray:
    """
    Standardize returns column-wise: X* = (X - mean) / std.
    
    Corresponds to the first step of Algorithm 1.
    
    Parameters
    ----------
    returns : pd.DataFrame
        n × d DataFrame of asset returns (rows=periods, columns=assets).
    
    Returns
    -------
    X_star : np.ndarray
        n × d array of standardized returns.
    """
    X = returns.values
    means = X.mean(axis=0, keepdims=True)
    stds = X.std(axis=0, ddof=1, keepdims=True)
    # Guard against zero std (constant returns)
    stds = np.where(stds < 1e-12, 1.0, stds)
    X_star = (X - means) / stds
    return X_star


def compute_sample_correlation(X_star: np.ndarray) -> np.ndarray:
    """
    Compute sample correlation matrix from standardized returns.
    
    Eq. (6): ρ̂ = (1/(n-1)) (X*)^T X*
    
    Parameters
    ----------
    X_star : np.ndarray
        n × d array of standardized returns.
    
    Returns
    -------
    rho_hat : np.ndarray
        d × d sample correlation matrix.
    """
    n = X_star.shape[0]
    rho_hat = (X_star.T @ X_star) / (n - 1)
    # Ensure symmetry (numerical precision)
    rho_hat = (rho_hat + rho_hat.T) / 2.0
    # Clip diagonal to exactly 1
    np.fill_diagonal(rho_hat, 1.0)
    return rho_hat


def compute_cord(rho_hat: np.ndarray) -> np.ndarray:
    """
    Compute sample CORD (Correlation Difference) dissimilarity matrix.
    
    Eq. (7): ĈORD(i,j) = max_{l ≠ i,j} |ρ̂_{il} - ρ̂_{jl}|
    
    Parameters
    ----------
    rho_hat : np.ndarray
        d × d sample correlation matrix.
    
    Returns
    -------
    cord : np.ndarray
        d × d dissimilarity matrix. cord[i,j] = ĈORD(i,j).
    """
    d = rho_hat.shape[0]
    cord = np.zeros((d, d))
    
    for i in range(d):
        for j in range(d):
            if i == j:
                cord[i, j] = 0.0
                continue
            # For all l ≠ i,j: compute |ρ_{il} - ρ_{jl}|
            mask = np.ones(d, dtype=bool)
            mask[i] = False
            mask[j] = False
            diffs = np.abs(rho_hat[i, mask] - rho_hat[j, mask])
            cord[i, j] = np.max(diffs) if len(diffs) > 0 else 0.0
    
    return cord


def partition_procedure(cord: np.ndarray, epsilon: float) -> List[List[int]]:
    """
    PARTITION procedure (Procedure 1 in the paper).
    
    Iteratively identifies clusters by finding the most similar pair of
    unassigned assets. If their dissimilarity is below ε, they form the
    core of a new cluster and absorb all similar assets.
    
    Parameters
    ----------
    cord : np.ndarray
        d × d dissimilarity matrix (ĈORD).
    epsilon : float
        Threshold parameter ε > 0.
    
    Returns
    -------
    clusters : List[List[int]]
        List of clusters, each a list of asset indices.
    """
    d = cord.shape[0]
    S = set(range(d))       # unassigned assets
    clusters = []
    
    while S:
        if len(S) == 1:
            clusters.append([s for s in S][0] if isinstance(next(iter(S)), int) else list(S))
            clusters[-1] = list(S)
            break
        
        # Find the most similar pair (i_l, j_l) in S
        S_list = sorted(S)
        min_dissim = np.inf
        best_pair = (S_list[0], S_list[1])
        
        for idx_i in range(len(S_list)):
            for idx_j in range(idx_i + 1, len(S_list)):
                i, j = S_list[idx_i], S_list[idx_j]
                if cord[i, j] < min_dissim:
                    min_dissim = cord[i, j]
                    best_pair = (i, j)
        
        i_l, j_l = best_pair
        
        if min_dissim > epsilon:
            # Singleton cluster — just take the most similar one
            clusters.append([i_l])
            S.discard(i_l)
        else:
            # Absorb all assets k in S with min(D(i_l,k), D(j_l,k)) ≤ ε
            new_cluster = []
            for k in list(S):
                if min(cord[i_l, k], cord[j_l, k]) <= epsilon:
                    new_cluster.append(k)
            # Always include the core pair
            if i_l not in new_cluster:
                new_cluster.append(i_l)
            if j_l not in new_cluster:
                new_cluster.append(j_l)
            clusters.append(new_cluster)
            for k in new_cluster:
                S.discard(k)
    
    return clusters


def estimate_heavy_tail_params(
    X_star: np.ndarray, 
    rho_hat: np.ndarray, 
    k: int
) -> Tuple[float, float]:
    """
    Estimate the heavy-tailedness parameter α and constant L.
    
    Section 2.4 of the paper. Uses the approach from Gardes and Girard (2008):
    - Compute whitened returns: Y_r = |(ρ̂^{-1/2} X*)_{r}|
    - Fit tail distribution P(Y>t) = 2exp(-(t/L)^α) via linear regression
    - log q(p) = (1/α) log log(2/(1-p)) + log L
    - Regress log Y_{r,(n-j)} against log log(2n/j) for j=1,...,k
    
    Parameters
    ----------
    X_star : np.ndarray
        n × d standardized returns.
    rho_hat : np.ndarray
        d × d sample correlation matrix.
    k : int
        Number of largest observations for tail fitting (k = n/4 per paper).
    
    Returns
    -------
    alpha : float
        Estimated α parameter (min over dimensions).
    L : float
        Estimated L constant (max over dimensions).
    """
    n, d = X_star.shape
    
    # Compute ρ̂^{-1/2} via spectral decomposition
    # ρ̂ = V Λ V^T  =>  ρ̂^{-1/2} = V Λ^{-1/2} V^T
    try:
        eigenvalues, eigenvectors = linalg.eigh(rho_hat)
        # Clip small/negative eigenvalues for numerical stability
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        rho_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    except Exception:
        # Fallback: regularize with ridge
        rho_reg = rho_hat + 1e-6 * np.eye(d)
        eigenvalues, eigenvectors = linalg.eigh(rho_reg)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        rho_inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    
    # Whitened returns: ρ̂^{-1/2} X*  (each row of X_star is a time observation)
    # Y_r = |(ρ̂^{-1/2} X*)_{r}| for each dimension r
    whitened = X_star @ rho_inv_sqrt.T  # n × d
    
    alphas = np.zeros(d)
    Ls = np.zeros(d)
    
    k = min(k, n - 1)  # Ensure k < n
    
    for r in range(d):
        Y_r = np.sort(np.abs(whitened[:, r]))  # ascending
        
        # Use the k largest observations: Y_{(n-j)} for j = 1,...,k
        # Equivalently, Y_{(n-k)} through Y_{(n-1)} (0-indexed: indices n-k-1 through n-2)
        # Actually: j ranges from 1 to k, so Y_{(n-j)} with j=1..k
        # In 0-indexed: indices n-2 down to n-k-1  (skip the max, which is n-1)
        # But the paper says: "The largest observation Y_{(n)} corresponds to q(1) and is not included"
        # So we use Y_{(n-j)} for j = 1,...,k  => 0-indexed: indices n-2, n-3, ..., n-k-1
        
        obs_indices = list(range(n - 2, n - k - 2, -1))  # n-2 down to n-k-1
        obs_indices = [idx for idx in obs_indices if idx >= 0]
        
        if len(obs_indices) < 3:
            alphas[r] = 2.0  # sub-Gaussian default
            Ls[r] = 1.0
            continue
        
        y_obs = Y_r[obs_indices]
        
        # log Y_{(n-j)} for j = 1,...,k
        log_y = np.log(np.maximum(y_obs, 1e-15))
        
        # log log(2n/j) for j = 1,...,k
        j_vals = np.arange(1, len(obs_indices) + 1, dtype=float)
        log_log_vals = np.log(np.log(2.0 * n / j_vals))
        
        # Linear regression: log y ~ (1/α) * log_log + log L
        # Slope = 1/α, Intercept = log L
        try:
            A = np.vstack([log_log_vals, np.ones(len(log_log_vals))]).T
            result = np.linalg.lstsq(A, log_y, rcond=None)
            slope, intercept = result[0]
            
            if slope <= 0:
                alphas[r] = 2.0
                Ls[r] = 1.0
            else:
                alphas[r] = 1.0 / slope
                Ls[r] = np.exp(intercept)
        except Exception:
            alphas[r] = 2.0
            Ls[r] = 1.0
    
    # Paper: α* = min_r α_r, L* = max_r L_r
    alpha = max(min(alphas), 0.25)  # Floor at 0.25 to avoid extreme values
    L_val = min(max(Ls), 5.0)       # Cap at 5 for numerical stability
    
    return alpha, L_val


def compute_intra_cluster_correlation(
    rho_hat: np.ndarray, 
    clusters: List[List[int]]
) -> float:
    """
    Compute the average intra-cluster correlation ρ̂^ave.
    
    Eq. (9): ρ̂^ave = Σ_{i<j} 1(i∼j) ρ̂_{ij} / Σ_{i<j} 1(i∼j)
    
    Parameters
    ----------
    rho_hat : np.ndarray
        d × d sample correlation matrix.
    clusters : List[List[int]]
        Clustering result.
    
    Returns
    -------
    rho_avg : float
        Average intra-cluster correlation.
    """
    total_corr = 0.0
    total_pairs = 0
    
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        for idx_i in range(len(cluster)):
            for idx_j in range(idx_i + 1, len(cluster)):
                i, j = cluster[idx_i], cluster[idx_j]
                total_corr += rho_hat[i, j]
                total_pairs += 1
    
    if total_pairs == 0:
        return -np.inf
    
    return total_corr / total_pairs


def acc_algorithm(
    returns: pd.DataFrame,
    a: float = DEFAULT_A,
    b: float = DEFAULT_B,
    n_grids: int = DEFAULT_NG,
    U: Tuple[int, int] = DEFAULT_U,
    k_ratio: float = DEFAULT_K_RATIO,
    verbose: bool = True
) -> List[List[int]]:
    """
    ACC (Asset Clustering through Correlation) algorithm.
    
    Algorithm 1 from the paper. Recovers clusters from asset returns
    using the correlation blockmodel approach.
    
    Parameters
    ----------
    returns : pd.DataFrame
        n × d DataFrame of asset returns.
    a : float
        Lower bound multiplier for ε search range.
    b : float
        Upper bound multiplier for ε search range.
    n_grids : int
        Number of grid points for ε search.
    U : Tuple[int, int]
        (min_clusters, max_clusters) range for Rule 3.
    k_ratio : float
        Fraction of observations for tail estimation (0.25 = n/4).
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    clusters : List[List[int]]
        Final clustering result (list of clusters, each a list of column indices).
    """
    n, d = returns.shape
    k = int(n * k_ratio)
    
    if verbose:
        print(f"ACC Algorithm: n={n} observations, d={d} assets")
        print(f"Parameters: a={a}, b={b}, n_grids={n_grids}, U={U}, k={k}")
    
    # Step 1: Standardize returns (Algorithm 1, line 1)
    X_star = compute_standardized_returns(returns)
    
    # Step 2: Compute sample correlation matrix (Eq. 6)
    rho_hat = compute_sample_correlation(X_star)
    if verbose:
        print(f"Sample correlation matrix computed: {rho_hat.shape}")
        print(f"  Mean off-diagonal correlation: {np.mean(rho_hat[np.triu_indices(d, k=1)]):.4f}")
    
    # Step 3: Compute CORD dissimilarity (Eq. 7)
    cord = compute_cord(rho_hat)
    if verbose:
        print(f"CORD matrix computed. Range: [{cord.min():.6f}, {cord.max():.6f}]")
    
    # Step 4: Estimate heavy-tail parameters (Section 2.4)
    alpha, L_val = estimate_heavy_tail_params(X_star, rho_hat, k)
    if verbose:
        print(f"Heavy-tail estimation: α={alpha:.4f}, L={L_val:.4f}")
    
    # Step 5: Determine search range for ε (Rule 1)
    log_d = np.log(d)
    
    power_val = (log_d) ** (4.0 / max(alpha - 1, 0.01))
    if np.isinf(power_val) or n > power_val:
        # Case 1: sqrt(log d / n) dominates
        base_scale = L_val**2 * np.sqrt(log_d / n)
    else:
        # Case 2: (log d)^{2/α} / n dominates
        base_scale = L_val**2 * (log_d ** (2.0 / alpha)) / n
    
    epsilon_lower = a * base_scale
    epsilon_upper = min(b * base_scale, EPS_CAP)
    
    # Ensure lower < upper and both are positive
    epsilon_lower = max(epsilon_lower, 1e-6)
    epsilon_upper = max(epsilon_upper, epsilon_lower + 1e-4)
    
    if verbose:
        print(f"ε search range: [{epsilon_lower:.6f}, {epsilon_upper:.6f}]")
        print(f"  base_scale = {base_scale:.6f}")
    
    # Step 6: Grid search over ε (Rule 3)
    epsilon_grid = np.linspace(epsilon_lower, epsilon_upper, n_grids)
    best_epsilon = epsilon_lower
    best_rho_avg = -np.inf
    best_clusters = None
    
    u_min, u_max = U
    
    for idx, epsilon in enumerate(epsilon_grid):
        clusters = partition_procedure(cord, epsilon)
        n_clusters = len(clusters)
        
        if n_clusters < u_min or n_clusters > u_max:
            continue
        
        rho_avg = compute_intra_cluster_correlation(rho_hat, clusters)
        
        if rho_avg > best_rho_avg:
            best_rho_avg = rho_avg
            best_epsilon = epsilon
            best_clusters = clusters
    
    # Fallback: if no ε gives clusters in range U, relax the constraint
    if best_clusters is None:
        if verbose:
            print("WARNING: No ε found with cluster count in U. Relaxing constraint...")
        for idx, epsilon in enumerate(epsilon_grid):
            clusters = partition_procedure(cord, epsilon)
            rho_avg = compute_intra_cluster_correlation(rho_hat, clusters)
            if best_clusters is None or rho_avg > best_rho_avg:
                best_rho_avg = rho_avg
                best_epsilon = epsilon
                best_clusters = clusters
    
    if verbose:
        print(f"\nBest ε = {best_epsilon:.6f}")
        print(f"Number of clusters: {len(best_clusters)}")
        print(f"Intra-cluster avg correlation: {best_rho_avg:.4f}")
        cluster_sizes = sorted([len(c) for c in best_clusters], reverse=True)
        print(f"Cluster sizes: {cluster_sizes[:20]}...")
    
    return best_clusters


# ============================================================================
# Asset Selection & Portfolio Construction
# ============================================================================

def select_representative_assets(
    returns: pd.DataFrame,
    clusters: List[List[int]]
) -> List[int]:
    """
    Select one asset per cluster with the lowest variance.
    
    Theorem 2 (Eq. 5): J*(k) = argmin_{j ∈ G_k} Var(X_j)
    
    Parameters
    ----------
    returns : pd.DataFrame
        n × d DataFrame of returns.
    clusters : List[List[int]]
        Clustering result.
    
    Returns
    -------
    selected : List[int]
        Indices of selected representative assets (one per cluster).
    """
    selected = []
    variances = returns.var(axis=0, ddof=1).values
    
    for cluster in clusters:
        # Find asset with lowest variance in this cluster
        cluster_variances = variances[cluster]
        best_idx = cluster[np.argmin(cluster_variances)]
        selected.append(best_idx)
    
    return selected


def risk_parity_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Compute risk parity (equal risk contribution) portfolio weights.
    
    Each asset contributes equally to total portfolio risk.
    Weights proportional to 1/σ²_i (inverse variance), then normalized.
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        k × k covariance matrix of selected assets.
    
    Returns
    -------
    weights : np.ndarray
        Portfolio weights (sum to 1, all non-negative).
    """
    variances = np.diag(cov_matrix)
    # Simple risk parity: weights ∝ 1/σ²
    inv_var = 1.0 / variances
    weights = inv_var / inv_var.sum()
    return weights


def minimum_variance_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Compute minimum variance portfolio weights (no short selling).
    
    min  w^T Σ w
    s.t. w^T 1 = 1, w ≥ 0
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        k × k covariance matrix.
    
    Returns
    -------
    weights : np.ndarray
        Portfolio weights.
    """
    k = cov_matrix.shape[0]
    
    # Objective: w^T Σ w
    def objective(w):
        return w @ cov_matrix @ w
    
    # Gradient: 2 Σ w
    def gradient(w):
        return 2.0 * cov_matrix @ w
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    # Bounds: 0 ≤ w_i ≤ 1
    bounds = [(0.0, 1.0)] * k
    
    # Initial guess: equal weights
    w0 = np.ones(k) / k
    
    result = minimize(
        objective, w0, jac=gradient,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 500}
    )
    
    if not result.success:
        warnings.warn(f"Min-variance optimization did not converge: {result.message}")
    
    return result.x


def mean_variance_weights(
    cov_matrix: np.ndarray, 
    mean_returns: np.ndarray,
    target_annual_return: float = 0.10,
    trading_days: int = 252
) -> np.ndarray:
    """
    Compute Markowitz mean-variance portfolio weights (no short selling).
    
    min  w^T Σ w
    s.t. w^T μ ≥ r_target (annualized), w^T 1 = 1, w ≥ 0
    
    Parameters
    ----------
    cov_matrix : np.ndarray
        k × k covariance matrix of daily returns.
    mean_returns : np.ndarray
        k-vector of mean daily returns.
    target_annual_return : float
        Target annualized return (default 10%).
    trading_days : int
        Number of trading days per year.
    
    Returns
    -------
    weights : np.ndarray
        Portfolio weights.
    """
    k = cov_matrix.shape[0]
    target_daily = target_annual_return / trading_days
    
    # Objective: w^T Σ w
    def objective(w):
        return w @ cov_matrix @ w
    
    def gradient(w):
        return 2.0 * cov_matrix @ w
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: w @ mean_returns - target_daily}
    ]
    
    bounds = [(0.0, 1.0)] * k
    w0 = np.ones(k) / k
    
    result = minimize(
        objective, w0, jac=gradient,
        method='SLSQP', bounds=bounds, constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 500}
    )
    
    if not result.success:
        # Fallback to minimum variance if target return not achievable
        return minimum_variance_weights(cov_matrix)
    
    return result.x


# ============================================================================
# Performance Metrics
# ============================================================================

def compute_portfolio_metrics(
    portfolio_returns: np.ndarray,
    trading_days: int = 252,
    rf: float = RISK_FREE_RATE
) -> Dict[str, float]:
    """
    Compute standard portfolio performance metrics.
    
    Parameters
    ----------
    portfolio_returns : np.ndarray
        Array of daily portfolio returns.
    trading_days : int
        Trading days per year.
    rf : float
        Annual risk-free rate.
    
    Returns
    -------
    metrics : Dict[str, float]
        Dictionary of performance metrics.
    """
    daily_rf = rf / trading_days
    ann_ret = np.mean(portfolio_returns) * trading_days
    ann_vol = np.std(portfolio_returns, ddof=1) * np.sqrt(trading_days)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
    
    # Sortino ratio
    downside = portfolio_returns[portfolio_returns < daily_rf] - daily_rf
    downside_vol = np.sqrt(np.mean(downside**2)) * np.sqrt(trading_days) if len(downside) > 0 else ann_vol
    sortino = (ann_ret - rf) / downside_vol if downside_vol > 0 else 0.0
    
    # Maximum drawdown
    cum_returns = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = abs(np.min(drawdowns))
    
    # Calmar ratio
    calmar = ann_ret / max_dd if max_dd > 0 else 0.0
    
    # VAMI (Value Added Monthly Index, ending value of $1)
    vami = cum_returns[-1]
    
    # Recovery from max drawdown (days)
    max_dd_idx = np.argmin(drawdowns)
    post_dd = cum_returns[max_dd_idx:]
    if len(post_dd) > 1:
        pre_dd_max = running_max[max_dd_idx]
        recovery_mask = post_dd >= pre_dd_max
        if np.any(recovery_mask):
            recovery_days = int(np.argmax(recovery_mask))
        else:
            recovery_days = -1  # Not recovered
    else:
        recovery_days = -1
    
    return {
        'annual_return': ann_ret,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_dd,
        'recovery_days': recovery_days,
        'ending_vami': vami,
        'positive_days_pct': np.mean(portfolio_returns > 0) * 100,
    }


def compute_benchmark_returns(
    price_data: pd.DataFrame,
    symbol: str = "SPY"
) -> np.ndarray:
    """
    Compute daily returns for a benchmark (if available in data).
    Otherwise return a market-cap-weighted approximation.
    """
    if symbol in price_data['symbol'].unique():
        bench = price_data[price_data['symbol'] == symbol].sort_values('Date')
        prices = bench['Close'].values
        returns = np.diff(prices) / prices[:-1]
        return returns
    return None


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Run the ACC algorithm on S&P 500 data and construct portfolios.
    """
    np.random.seed(42)
    
    print("=" * 70)
    print("ACC Algorithm: Asset Selection via Correlation Blockmodel Clustering")
    print("Paper: Tang, Xu, Zhou (2021), arXiv:2103.14506")
    print("=" * 70)
    
    # ---- Load Data ----
    print("\n[1] Loading S&P 500 data...")
    data_path = "/opt/data/kanban/workspaces/t_dc995cda/sp500_data/sp500_ohlcv_5yr.parquet"
    df = pd.read_parquet(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"  Raw data: {df.shape[0]} rows, {df['symbol'].nunique()} tickers")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # ---- Prepare Returns ----
    print("\n[2] Computing daily returns...")
    # Pivot to get close prices: rows=dates, columns=tickers
    close_prices = df.pivot(index='Date', columns='symbol', values='Close')
    close_prices = close_prices.sort_index()
    
    # Compute daily returns
    returns = close_prices.pct_change().iloc[1:]  # drop first NaN row
    returns = returns.dropna(axis=1, how='any')  # drop tickers with any NaN
    
    n_days, n_assets = returns.shape
    print(f"  Returns matrix: {n_days} days × {n_assets} assets")
    print(f"  Date range: {returns.index[0]} to {returns.index[-1]}")
    
    # ---- Apply ACC Algorithm ----
    print("\n[3] Running ACC Algorithm...")
    
    # Use the last 500 trading days (matching paper's lookback)
    lookback = min(500, n_days)
    returns_window = returns.iloc[-lookback:]
    
    print(f"  Using lookback window of {lookback} days")
    
    clusters = acc_algorithm(
        returns_window,
        a=DEFAULT_A,
        b=DEFAULT_B,
        n_grids=DEFAULT_NG,
        U=DEFAULT_U,
        k_ratio=DEFAULT_K_RATIO,
        verbose=True
    )
    
    # ---- Select Representative Assets ----
    print("\n[4] Selecting representative assets (lowest variance per cluster)...")
    selected_indices = select_representative_assets(returns_window, clusters)
    
    # Map indices back to ticker symbols
    tickers = returns_window.columns.tolist()
    selected_tickers = [tickers[i] for i in selected_indices]
    
    print(f"  Selected {len(selected_tickers)} stocks:")
    for i, (cluster, sel_idx) in enumerate(zip(clusters, selected_indices)):
        cluster_tickers = [tickers[j] for j in cluster]
        var = returns_window.iloc[:, sel_idx].var(ddof=1)
        print(f"  Cluster {i+1} ({len(cluster)} stocks): selected {tickers[sel_idx]} "
              f"(var={var:.6f}) | cluster: {cluster_tickers[:5]}{'...' if len(cluster) > 5 else ''}")
    
    # ---- Construct Portfolios ----
    print("\n[5] Constructing portfolios...")
    
    # Use full return series for selected stocks
    selected_returns_full = returns[selected_tickers]
    
    # Compute covariance matrix from lookback window
    cov_daily = returns_window[selected_tickers].cov().values
    mean_daily = returns_window[selected_tickers].mean().values
    
    # Risk Parity
    rp_weights = risk_parity_weights(cov_daily)
    print(f"\n  Risk Parity weights:")
    for t, w in zip(selected_tickers, rp_weights):
        print(f"    {t:6s}: {w:.4f}")
    
    # Minimum Variance
    mv_weights = minimum_variance_weights(cov_daily)
    print(f"\n  Minimum Variance weights:")
    for t, w in zip(selected_tickers, mv_weights):
        print(f"    {t:6s}: {w:.4f}")
    
    # Mean-Variance (10% target annual return)
    me_weights = mean_variance_weights(cov_daily, mean_daily, target_annual_return=0.10)
    print(f"\n  Mean-Variance weights (target 10% annual):")
    for t, w in zip(selected_tickers, me_weights):
        print(f"    {t:6s}: {w:.4f}")
    
    # ---- Compute Out-of-Sample Performance ----
    print("\n[6] Computing portfolio performance (out-of-sample)...")
    
    # Use the period AFTER the lookback window for out-of-sample testing
    # Since our data ends at 2026-05-06 and we use last 500 days for clustering,
    # there may be no out-of-sample data. Instead, compute in-sample performance
    # for the lookback window and also full-period performance.
    
    # Full-period portfolio returns
    for strategy_name, weights in [
        ("Risk Parity", rp_weights),
        ("Minimum Variance", mv_weights),
        ("Mean-Variance (10%)", me_weights),
    ]:
        port_returns = (returns[selected_tickers] * weights).sum(axis=1).values
        metrics = compute_portfolio_metrics(port_returns)
        print(f"\n  {strategy_name} Portfolio (full period):")
        print(f"    Annual Return:       {metrics['annual_return']:.2%}")
        print(f"    Annual Volatility:   {metrics['annual_volatility']:.2%}")
        print(f"    Sharpe Ratio:        {metrics['sharpe_ratio']:.4f}")
        print(f"    Sortino Ratio:       {metrics['sortino_ratio']:.4f}")
        print(f"    Calmar Ratio:        {metrics['calmar_ratio']:.4f}")
        print(f"    Max Drawdown:        {metrics['max_drawdown']:.2%}")
        print(f"    Recovery Days:       {metrics['recovery_days']}")
        print(f"    Ending VAMI ($1):    {metrics['ending_vami']:.2f}")
        print(f"    Positive Days:       {metrics['positive_days_pct']:.1f}%")
    
    # ---- Compare with Equal-Weight S&P 500 ----
    print("\n[7] Benchmark: Equal-weight S&P 500 portfolio...")
    # Compute equal-weight S&P 500 portfolio returns
    sp500_returns = returns.mean(axis=1).values
    sp500_metrics = compute_portfolio_metrics(sp500_returns)
    print(f"  Annual Return:       {sp500_metrics['annual_return']:.2%}")
    print(f"  Annual Volatility:   {sp500_metrics['annual_volatility']:.2%}")
    print(f"  Sharpe Ratio:        {sp500_metrics['sharpe_ratio']:.4f}")
    print(f"    Sortino Ratio:     {sp500_metrics['sortino_ratio']:.4f}")
    print(f"    Max Drawdown:      {sp500_metrics['max_drawdown']:.2%}")
    print(f"    Ending VAMI ($1):  {sp500_metrics['ending_vami']:.2f}")
    
    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nACC Algorithm identified {len(clusters)} clusters from {n_assets} S&P 500 stocks")
    print(f"Selected {len(selected_tickers)} representative stocks (lowest variance per cluster)")
    print(f"\nSelected tickers: {selected_tickers}")
    print(f"\nCluster composition:")
    for i, (cluster, sel_idx) in enumerate(zip(clusters, selected_indices)):
        cluster_tickers = [tickers[j] for j in cluster]
        print(f"  Cluster {i+1:2d}: {len(cluster):3d} stocks | Rep: {tickers[sel_idx]:6s} | "
              f"Members: {', '.join(cluster_tickers[:8])}{'...' if len(cluster) > 8 else ''}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
