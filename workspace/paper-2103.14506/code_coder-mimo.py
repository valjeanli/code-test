"""
Implementation of: Asset Selection via Correlation Blockmodel Clustering
Paper: arXiv:2103.14506 (Tang, Xu, Zhou 2021)

ACC (Asset Clustering through Correlation) Algorithm:
  - Correlation blockmodel for asset clustering
  - CORD dissimilarity measure
  - Grid-search PARTITION procedure
  - Portfolio construction: Risk Parity, Min Variance, Mean-Variance
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Data Loading & Preprocessing
# ─────────────────────────────────────────────────────────────

def load_data(parquet_path: str, lookback_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load S&P 500 OHLCV data and compute daily returns.

    Args:
        parquet_path: Path to parquet file with columns [date, symbol, open, high, low, close, volume]
        lookback_days: Number of trading days to use for the analysis window

    Returns:
        prices: DataFrame of close prices (dates x symbols)
        returns: DataFrame of daily simple returns (dates x symbols)
    """
    df = pd.read_parquet(parquet_path)
    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    # Remove timezone info from date for consistent indexing
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    # Pivot to wide format: dates as index, symbols as columns
    prices = df.pivot(index="date", columns="symbol", values="close").sort_index()
    # Drop any symbols with missing data
    prices = prices.dropna(axis=1)
    # Compute simple daily returns
    returns = prices.pct_change().dropna()
    # Use the last lookback_days
    if len(returns) > lookback_days:
        returns = returns.iloc[-lookback_days:]
        prices = prices.iloc[-lookback_days:]
    return prices, returns


def standardize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Standardize returns: (X - mean) / std per asset. Eq. in Section 2.1."""
    return (returns - returns.mean()) / returns.std()


# ─────────────────────────────────────────────────────────────
# SECTION 2: Correlation & CORD Matrix
# ─────────────────────────────────────────────────────────────

def compute_correlation_matrix(std_returns: pd.DataFrame) -> np.ndarray:
    """Compute sample correlation matrix rho_hat. Eq. 2."""
    return std_returns.corr().values


def compute_cord_matrix(rho: np.ndarray) -> np.ndarray:
    """Compute CORD dissimilarity matrix.

    CORD(i, j) = max_{l != i,j} |rho_{il} - rho_{jl}|   (Eq. 4)

    Args:
        rho: d x d correlation matrix

    Returns:
        d x d CORD matrix (symmetric, zero diagonal)
    """
    d = rho.shape[0]
    cord = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            # For all l != i, j, compute |rho_{il} - rho_{jl}|
            max_diff = 0.0
            for l in range(d):
                if l == i or l == j:
                    continue
                diff = abs(rho[i, l] - rho[j, l])
                if diff > max_diff:
                    max_diff = diff
            cord[i, j] = max_diff
            cord[j, i] = max_diff
    return cord


def compute_cord_matrix_fast(rho: np.ndarray) -> np.ndarray:
    """Vectorized CORD computation. Same result, much faster for large d.

    CORD(i,j) = max_{l not in {i,j}} |rho[i,l] - rho[j,l]|
    """
    d = rho.shape[0]
    cord = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            # Compute |rho[i,:] - rho[j,:]|
            diff = np.abs(rho[i, :] - rho[j, :])
            # Exclude positions i and j
            mask = np.ones(d, dtype=bool)
            mask[i] = False
            mask[j] = False
            cord[i, j] = np.max(diff[mask])
            cord[j, i] = cord[i, j]
    return cord


# ─────────────────────────────────────────────────────────────
# SECTION 3: Hill Estimator for Tail Index
# ─────────────────────────────────────────────────────────────

def estimate_tail_index(std_returns: pd.DataFrame, k_frac: float = 0.1) -> float:
    """Estimate tail index alpha using Hill's estimator.

    The tail index governs the grid bounds for epsilon in the ACC algorithm.
    For Gaussian data, alpha = 2. For heavy-tailed financial data, alpha < 2.

    Args:
        std_returns: Standardized returns DataFrame
        k_frac: Fraction of observations to use for Hill estimator

    Returns:
        Estimated tail index alpha
    """
    # Flatten all standardized returns
    values = np.abs(std_returns.values.flatten())
    values = values[values > 0]  # avoid log(0)
    values = np.sort(values)[::-1]  # descending order

    n = len(values)
    k = max(int(n * k_frac), 10)
    k = min(k, n - 1)

    # Hill estimator: 1/alpha = (1/k) * sum_{i=1}^{k} log(Y_(i) / Y_(k+1))
    log_ratios = np.log(values[:k] / values[k])
    hill_estimate = np.mean(log_ratios)

    if hill_estimate > 0:
        alpha = 1.0 / hill_estimate
    else:
        alpha = 2.0  # default to Gaussian
    return alpha


# ─────────────────────────────────────────────────────────────
# SECTION 4: PARTITION Procedure (Algorithm 2 in paper)
# ─────────────────────────────────────────────────────────────

def partition(cord: np.ndarray, epsilon: float) -> List[List[int]]:
    """PARTITION procedure from the paper.

    Groups assets based on CORD dissimilarity with threshold epsilon.

    Algorithm:
    1. Start with all assets unassigned
    2. Pick a seed asset i_t
    3. Find its nearest neighbor j_t by CORD
    4. If CORD(i_t, j_t) > epsilon: i_t is a singleton
    5. Else: form cluster from all assets with CORD <= epsilon to i_t or j_t
    6. Repeat until all assets assigned

    Args:
        cord: d x d CORD dissimilarity matrix
        epsilon: Clustering threshold

    Returns:
        List of clusters, each cluster is a list of asset indices
    """
    d = cord.shape[0]
    remaining = set(range(d))
    clusters = []

    while remaining:
        # Pick seed (smallest index in remaining)
        i_t = min(remaining)

        # Find nearest neighbor in remaining set
        min_dist = float("inf")
        j_t = None
        for j in remaining:
            if j == i_t:
                continue
            if cord[i_t, j] < min_dist:
                min_dist = cord[i_t, j]
                j_t = j

        if j_t is None or min_dist > epsilon:
            # Singleton cluster
            clusters.append([i_t])
            remaining.remove(i_t)
        else:
            # Build cluster: all assets with CORD <= epsilon to i_t or j_t
            cluster = set()
            for k in list(remaining):
                if cord[i_t, k] <= epsilon or cord[j_t, k] <= epsilon:
                    cluster.add(k)
            # Must include at least i_t and j_t
            cluster.add(i_t)
            cluster.add(j_t)
            clusters.append(sorted(cluster))
            remaining -= cluster

    return clusters


# ─────────────────────────────────────────────────────────────
# SECTION 5: ACC Algorithm (Algorithm 1 in paper)
# ─────────────────────────────────────────────────────────────

def acc_algorithm(
    returns: pd.DataFrame,
    n_grid: int = 20,
    target_clusters_min: int = 15,
    target_clusters_max: int = 25,
) -> Dict:
    """Full ACC (Asset Clustering through Correlation) algorithm.

    Stage 1: Preparation
      - Standardize returns
      - Compute correlation matrix and CORD matrix
      - Estimate tail index

    Stage 2: Grid Search
      - Search over epsilon values
      - Run PARTITION for each epsilon
      - Select epsilon that yields desired cluster count

    Args:
        returns: DataFrame of daily returns (dates x assets)
        n_grid: Number of epsilon values to search
        target_clusters_min: Minimum desired clusters
        target_clusters_max: Maximum desired clusters

    Returns:
        Dictionary with:
          - clusters: List of clusters (each is list of asset indices)
          - epsilon: Selected epsilon value
          - n_clusters: Number of clusters
          - cord_matrix: CORD dissimilarity matrix
          - corr_matrix: Correlation matrix
          - std_returns: Standardized returns
          - alpha: Estimated tail index
          - grid_results: Results for all grid points
    """
    n_assets = returns.shape[1]
    n_obs = returns.shape[0]
    symbols = returns.columns.tolist()

    print(f"  Assets: {n_assets}, Observations: {n_obs}")

    # Stage 1: Preparation
    print("  [Stage 1] Computing standardized returns...")
    std_returns = standardize_returns(returns)

    print("  [Stage 1] Computing correlation matrix...")
    rho = compute_correlation_matrix(std_returns)

    print("  [Stage 1] Computing CORD matrix...")
    cord = compute_cord_matrix_fast(rho)

    print("  [Stage 1] Estimating tail index (Hill estimator)...")
    alpha = estimate_tail_index(std_returns)
    print(f"    Tail index alpha = {alpha:.3f}")

    # Stage 2: Grid Search
    # Grid bounds: epsilon in [epsilon_min, epsilon_max]
    # epsilon_min based on Theorem 3: c1 * sqrt(log(d) / n)
    epsilon_min = np.sqrt(np.log(n_assets) / n_obs)
    epsilon_max = np.percentile(cord[np.triu_indices(n_assets, k=1)], 75)
    # Ensure bounds are sensible
    epsilon_min = max(epsilon_min, 0.01)
    epsilon_max = min(epsilon_max, 0.8)

    grid = np.linspace(epsilon_min, epsilon_max, n_grid)
    grid_results = []

    print(f"  [Stage 2] Grid search: epsilon in [{epsilon_min:.4f}, {epsilon_max:.4f}], {n_grid} points")
    for eps in grid:
        clusters = partition(cord, eps)
        k = len(clusters)
        sizes = [len(c) for c in clusters]
        grid_results.append({
            "epsilon": eps,
            "n_clusters": k,
            "cluster_sizes": sizes,
            "max_cluster_size": max(sizes),
            "min_cluster_size": min(sizes),
        })

    # Select epsilon: prefer cluster count in [target_min, target_max]
    # If none in range, pick the one closest to target_clusters_max
    selected = None
    for gr in grid_results:
        if target_clusters_min <= gr["n_clusters"] <= target_clusters_max:
            selected = gr
            break

    if selected is None:
        # Pick closest to target
        selected = min(
            grid_results,
            key=lambda g: abs(g["n_clusters"] - target_clusters_max)
        )

    eps_star = selected["epsilon"]
    clusters = partition(cord, eps_star)

    # Also compute within-cluster average correlation for validation
    cluster_avg_corr = []
    for cl in clusters:
        if len(cl) > 1:
            sub = rho[np.ix_(cl, cl)]
            # Average off-diagonal correlation
            mask = ~np.eye(len(cl), dtype=bool)
            avg = sub[mask].mean()
            cluster_avg_corr.append(avg)
        else:
            cluster_avg_corr.append(np.nan)

    print(f"  [Result] epsilon* = {eps_star:.4f}, K = {len(clusters)} clusters")
    print(f"    Cluster sizes: {[len(c) for c in clusters]}")
    print(f"    Avg within-cluster correlation: {np.nanmean(cluster_avg_corr):.4f}")

    return {
        "clusters": clusters,
        "epsilon": eps_star,
        "n_clusters": len(clusters),
        "cord_matrix": cord,
        "corr_matrix": rho,
        "std_returns": std_returns,
        "alpha": alpha,
        "grid_results": grid_results,
        "cluster_avg_corr": cluster_avg_corr,
        "symbols": symbols,
    }


# ─────────────────────────────────────────────────────────────
# SECTION 6: Representative Stock Selection
# ─────────────────────────────────────────────────────────────

def select_representatives(
    clusters: List[List[int]],
    returns: pd.DataFrame,
    method: str = "min_volatility",
) -> List[int]:
    """Select one representative stock per cluster.

    Per the paper (Theorem 2), any stock in a cluster is interchangeable
    in terms of correlations with others. So we select by idiosyncratic
    properties: minimum volatility or maximum Sharpe ratio.

    Args:
        clusters: List of clusters (each is list of asset indices)
        returns: Daily returns DataFrame
        method: 'min_volatility' or 'max_sharpe'

    Returns:
        List of selected asset indices (one per cluster)
    """
    vol = returns.std()
    mean_ret = returns.mean()
    selected = []

    for cl in clusters:
        cl_symbols = [returns.columns[i] for i in cl]
        if method == "min_volatility":
            vols = vol[cl_symbols]
            best_idx = cl[vols.values.argmin()]
        elif method == "max_sharpe":
            # Annualized Sharpe (risk-free = 0)
            sharpes = (mean_ret[cl_symbols] * 252) / (vol[cl_symbols] * np.sqrt(252))
            best_idx = cl[sharpes.values.argmax()]
        else:
            best_idx = cl[0]
        selected.append(best_idx)

    return selected


# ─────────────────────────────────────────────────────────────
# SECTION 7: Portfolio Allocation Strategies
# ─────────────────────────────────────────────────────────────

def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Risk parity portfolio: equalize marginal risk contributions.

    Eq. 17-20 in the paper.
    σ_i(w) = w_i * (Σw)_i / σ(w)
    Target: σ_i(w) = σ(w) / d for all i

    Args:
        cov: d x d covariance matrix

    Returns:
        Weight vector (sums to 1, all non-negative)
    """
    d = cov.shape[0]
    w0 = np.ones(d) / d

    def objective(w):
        sigma = np.sqrt(w @ cov @ w)
        marginal_contrib = (cov @ w)
        risk_contrib = w * marginal_contrib / sigma
        target = sigma / d
        return np.sum((risk_contrib - target) ** 2)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * d

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return np.ones(d) / d


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """Minimum variance portfolio.

    Eq. 22: min w^T Σ w  s.t. w^T 1 = 1, w >= 0

    Args:
        cov: d x d covariance matrix

    Returns:
        Weight vector
    """
    d = cov.shape[0]
    w0 = np.ones(d) / d

    def objective(w):
        return w @ cov @ w

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * d

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        return np.ones(d) / d


def mean_variance_weights(cov: np.ndarray, mu: np.ndarray, target_return: float = 0.10) -> np.ndarray:
    """Markowitz mean-variance portfolio.

    Eq. 21: min w^T Σ w  s.t. w^T μ >= α, w^T 1 = 1, w >= 0

    Args:
        cov: d x d covariance matrix
        mu: Expected return vector (annualized)
        target_return: Target annual return (alpha)

    Returns:
        Weight vector
    """
    d = cov.shape[0]
    w0 = np.ones(d) / d

    def objective(w):
        return w @ cov @ w

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: w @ mu - target_return},
    ]
    bounds = [(0.0, 1.0)] * d

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        w = result.x
        w = np.maximum(w, 0)
        w /= w.sum()
        return w
    else:
        # Fall back to min variance if target return is infeasible
        return min_variance_weights(cov)


# ─────────────────────────────────────────────────────────────
# SECTION 8: Backtesting Framework
# ─────────────────────────────────────────────────────────────

def backtest(
    prices: pd.DataFrame,
    lookback: int = 252,
    rebalance_freq: int = 21,
    target_clusters: Tuple[int, int] = (15, 25),
    alloc_method: str = "risk_parity",
    rep_method: str = "min_volatility",
) -> pd.DataFrame:
    """Walk-forward backtest of ACC-based portfolio construction.

    At each rebalance point:
    1. Run ACC on the last `lookback` days of returns
    2. Select one representative stock per cluster
    3. Compute portfolio weights using the chosen allocation strategy
    4. Hold until next rebalance

    Args:
        prices: Price DataFrame (dates x symbols)
        lookback: Lookback window in trading days
        rebalance_freq: Rebalance every N trading days
        target_clusters: (min, max) target cluster count
        alloc_method: 'risk_parity', 'min_variance', or 'mean_variance'
        rep_method: 'min_volatility' or 'max_sharpe'

    Returns:
        DataFrame with daily portfolio values and metrics
    """
    returns = prices.pct_change().dropna()
    dates = returns.index
    n_dates = len(dates)
    symbols = returns.columns.tolist()
    n_assets = len(symbols)

    # Start after enough history
    start_idx = lookback
    portfolio_values = [1.0]
    portfolio_dates = [dates[start_idx]]
    current_weights = None

    rebalance_points = list(range(start_idx, n_dates, rebalance_freq))
    print(f"  Backtest: {n_dates} days, lookback={lookback}, rebalance every {rebalance_freq} days")
    print(f"  Rebalance points: {len(rebalance_points)}")
    print(f"  Allocation: {alloc_method}, Representative: {rep_method}")

    for rb_idx, rb_point in enumerate(rebalance_points):
        # Get lookback window
        window_start = max(0, rb_point - lookback)
        window_returns = returns.iloc[window_start:rb_point]

        # Run ACC
        acc_result = acc_algorithm(
            window_returns,
            target_clusters_min=target_clusters[0],
            target_clusters_max=target_clusters[1],
        )

        # Select representatives
        selected_indices = select_representatives(
            acc_result["clusters"], window_returns, method=rep_method
        )
        selected_symbols = [symbols[i] for i in selected_indices]

        # Compute covariance and expected returns for selected stocks
        sel_returns = window_returns[selected_symbols]
        cov = sel_returns.cov().values * 252  # annualized
        mu = sel_returns.mean().values * 252  # annualized

        # Compute weights
        if alloc_method == "risk_parity":
            weights = risk_parity_weights(cov)
        elif alloc_method == "min_variance":
            weights = min_variance_weights(cov)
        elif alloc_method == "mean_variance":
            weights = mean_variance_weights(cov, mu, target_return=0.10)
        else:
            weights = np.ones(len(selected_symbols)) / len(selected_symbols)

        current_weights = dict(zip(selected_symbols, weights))

        # Compute returns until next rebalance
        next_rb = rebalance_points[rb_idx + 1] if rb_idx + 1 < len(rebalance_points) else n_dates
        for t in range(rb_point, next_rb):
            day_return = 0.0
            for sym, w in current_weights.items():
                day_return += w * returns.iloc[t][sym]
            portfolio_values.append(portfolio_values[-1] * (1 + day_return))
            portfolio_dates.append(dates[t])

        if (rb_idx + 1) % 5 == 0 or rb_idx == 0:
            n_clust = acc_result["n_clusters"]
            print(f"    Rebalance {rb_idx+1}/{len(rebalance_points)}: "
                  f"K={n_clust}, selected={selected_symbols[:5]}..., "
                  f"NAV={portfolio_values[-1]:.4f}")

    # Build result DataFrame
    result = pd.DataFrame({
        "date": portfolio_dates,
        "nav": portfolio_values,
    }).set_index("date")

    # Compute benchmark (equal-weight all assets)
    benchmark_returns = returns.iloc[start_idx - 1:].copy()
    benchmark_nav = (1 + benchmark_returns.mean(axis=1)).cumprod().values
    # Align lengths
    min_len = min(len(result), len(benchmark_nav))
    result = result.iloc[:min_len].copy()
    result["benchmark_nav"] = benchmark_nav[:min_len]

    return result


# ─────────────────────────────────────────────────────────────
# SECTION 9: Performance Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(nav_series: pd.Series, annual_factor: float = 252) -> Dict:
    """Compute portfolio performance metrics.

    Args:
        nav_series: Series of net asset values
        annual_factor: Trading days per year

    Returns:
        Dictionary of performance metrics
    """
    returns = nav_series.pct_change().dropna()
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    n_years = len(returns) / annual_factor
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = returns.std() * np.sqrt(annual_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    cummax = nav_series.cummax()
    drawdown = (nav_series - cummax) / cummax
    max_dd = drawdown.min()

    # Sortino ratio
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(annual_factor) if len(downside) > 0 else 0.0
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
    }


# ─────────────────────────────────────────────────────────────
# SECTION 10: Main Execution
# ─────────────────────────────────────────────────────────────

def main():
    """Run the full ACC pipeline on S&P 500 data."""
    import os

    # Paths
    data_path = "/opt/data/kanban/workspaces/t_dc995cda/sp500_data/sp500_ohlcv_5yr.parquet"
    output_dir = "/opt/data/kanban/workspaces/t_265dd25c/paper-2103.14506"
    output_path = os.path.join(output_dir, "output_coder-mimo.txt")

    os.makedirs(output_dir, exist_ok=True)

    output_lines = []

    def log(msg):
        print(msg)
        output_lines.append(msg)

    log("=" * 70)
    log("ACC Algorithm Implementation - Paper 2103.14506")
    log("Asset Selection via Correlation Blockmodel Clustering")
    log("Tang, Xu, Zhou (2021)")
    log("=" * 70)

    # ── Load Data ──
    log("\n[1] LOADING DATA")
    log("-" * 40)
    prices, returns = load_data(data_path, lookback_days=1258)  # ~5 years
    log(f"  Date range: {returns.index[0]} to {returns.index[-1]}")
    log(f"  Trading days: {len(returns)}")
    log(f"  Stocks: {returns.shape[1]}")

    # ── Single ACC Run (on full dataset for analysis) ──
    log("\n[2] ACC ALGORITHM - FULL DATASET ANALYSIS")
    log("-" * 40)
    acc_result = acc_algorithm(returns, n_grid=20, target_clusters_min=15, target_clusters_max=25)

    log(f"\n  Selected epsilon: {acc_result['epsilon']:.4f}")
    log(f"  Number of clusters: {acc_result['n_clusters']}")
    log(f"  Tail index (alpha): {acc_result['alpha']:.3f}")

    log("\n  Grid search results:")
    log(f"  {'Epsilon':>10} {'Clusters':>10} {'Max Size':>10} {'Min Size':>10}")
    for gr in acc_result["grid_results"]:
        log(f"  {gr['epsilon']:10.4f} {gr['n_clusters']:10d} {gr['max_cluster_size']:10d} {gr['min_cluster_size']:10d}")

    log("\n  Cluster details:")
    symbols = acc_result["symbols"]
    for i, cl in enumerate(acc_result["clusters"]):
        cl_symbols = [symbols[idx] for idx in cl]
        avg_corr = acc_result["cluster_avg_corr"][i]
        log(f"  Cluster {i+1} (size={len(cl)}, avg_corr={avg_corr:.4f}): {cl_symbols}")

    # ── Representative Selection ──
    log("\n[3] REPRESENTATIVE STOCK SELECTION")
    log("-" * 40)
    selected_idx = select_representatives(acc_result["clusters"], returns, method="min_volatility")
    selected_syms = [symbols[i] for i in selected_idx]
    log(f"  Selected {len(selected_syms)} stocks (min volatility per cluster):")
    log(f"  {selected_syms}")

    # ── Portfolio Construction (single period) ──
    log("\n[4] PORTFOLIO CONSTRUCTION (Single Period)")
    log("-" * 40)
    sel_returns = returns[selected_syms]
    cov_ann = sel_returns.cov().values * 252
    mu_ann = sel_returns.mean().values * 252

    for method_name, method_func in [
        ("Risk Parity", risk_parity_weights),
        ("Min Variance", min_variance_weights),
    ]:
        w = method_func(cov_ann)
        log(f"\n  {method_name} Weights:")
        for sym, weight in zip(selected_syms, w):
            if weight > 0.001:
                log(f"    {sym:>6s}: {weight:.4f} ({weight*100:.1f}%)")
        port_vol = np.sqrt(w @ cov_ann @ w)
        port_ret = w @ mu_ann
        log(f"    Portfolio Return: {port_ret*100:.2f}%")
        log(f"    Portfolio Vol:    {port_vol*100:.2f}%")
        log(f"    Sharpe Ratio:    {port_ret/port_vol:.3f}")

    # ── Backtest ──
    log("\n[5] BACKTESTING")
    log("-" * 40)

    for alloc in ["risk_parity", "min_variance"]:
        log(f"\n  --- {alloc.upper()} ---")
        bt_result = backtest(
            prices,
            lookback=252,
            rebalance_freq=63,  # quarterly
            target_clusters=(15, 25),
            alloc_method=alloc,
            rep_method="min_volatility",
        )

        # Align for metrics
        min_len = min(len(bt_result), len(bt_result.dropna()))
        bt_clean = bt_result.dropna()

        metrics = compute_metrics(bt_clean["nav"])
        bench_metrics = compute_metrics(bt_clean["benchmark_nav"])

        log(f"\n  {alloc} Portfolio Metrics:")
        log(f"    Total Return:       {metrics['total_return']*100:>8.2f}%")
        log(f"    Ann. Return:        {metrics['annualized_return']*100:>8.2f}%")
        log(f"    Ann. Volatility:    {metrics['annualized_volatility']*100:>8.2f}%")
        log(f"    Sharpe Ratio:       {metrics['sharpe_ratio']:>8.3f}")
        log(f"    Sortino Ratio:      {metrics['sortino_ratio']:>8.3f}")
        log(f"    Max Drawdown:       {metrics['max_drawdown']*100:>8.2f}%")
        log(f"    Calmar Ratio:       {metrics['calmar_ratio']:>8.3f}")

        log(f"\n  Equal-Weight Benchmark:")
        log(f"    Total Return:       {bench_metrics['total_return']*100:>8.2f}%")
        log(f"    Ann. Return:        {bench_metrics['annualized_return']*100:>8.2f}%")
        log(f"    Sharpe Ratio:       {bench_metrics['sharpe_ratio']:>8.3f}")
        log(f"    Max Drawdown:       {bench_metrics['max_drawdown']*100:>8.2f}%")

    # ── Summary ──
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Paper: arXiv:2103.14506 - Asset Selection via Correlation Blockmodel")
    log(f"  Data: S&P 500, {returns.shape[1]} stocks, {returns.index[0]} to {returns.index[-1]}")
    log(f"  ACC found {acc_result['n_clusters']} clusters (epsilon={acc_result['epsilon']:.4f})")
    log(f"  Selected {len(selected_syms)} representative stocks")
    log(f"  Tail index alpha={acc_result['alpha']:.3f} (2.0=Gaussian)")
    log("=" * 70)

    # ── Save Output ──
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))
    log(f"\n  Output saved to: {output_path}")


if __name__ == "__main__":
    main()
