#!/usr/bin/env python3
"""
Regime Detection via Signature MMD (Auto-Evaluator)
====================================================
Implementation of the algorithm from:
  Horvath & Issa (2023) — "Non-parametric online market regime detection
  and regime clustering for multidimensional and path-dependent data structures"
  arXiv:2306.15835 / SSRN 3947905

Uses truncated path signatures (level 2) with RBF kernel MMD to detect
regime changes in S&P 500 daily price data. Outputs conform to the
output spec v1.0 from the evaluation plan.

References:
  - Eq. 16: Unbiased MMD estimator
  - Eq. 29: L-lag auto evaluation score
  - Definition 2.1: Path signature
  - Section 3.2.2: Auto-evaluator (non-parametric mode)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import json
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION — Hyperparameters from the paper (Section 6.1: Real Data)
# ==============================================================================

CONFIG = {
    # Data parameters
    "ticker": "^GSPC",
    "data_start": "2006-01-01",
    "data_end": "2026-01-01",
    "eval_start": "2021-05-10",
    "eval_end": "2026-05-06",

    # Path extraction (Definition 3.1)
    # h1 = sub-path length in trading days, h2 = ensemble size
    # Paper uses (8,8) for daily but (21,12) for monthly granularity
    "h1": 21,              # ~1 month of trading days per sub-path
    "h2": 12,              # 12 sub-paths per ensemble (~1 year context)

    # Path transformation (Section 3.1)
    "annualize_scale": np.sqrt(252),  # λ = dt^(-1/2) for daily data

    # Signature computation
    "sig_level": 2,         # Truncation level (1=increments, 2=area, 3=higher-order)

    # MMD parameters
    "kernel": "rbf",       # RBF-lifted signature kernel (Remark 2.19)
    "rbf_sigma": 1.0,      # σ in the RBF kernel

    # Auto-evaluator (Definition 3.9)
    "L": [4, 8, 12],       # Multi-scale lag set (compare 1, 2, 3 months ago)
    "L_weights": [0.5, 0.3, 0.2],  # w_l weights for score aggregation

    # Detection parameters
    "alpha": 0.95,          # Confidence level for critical threshold
    "prior_window": 200,    # Number of historical MMD scores for null distribution

    # Clustering (Section 3.3)
    "n_regimes_max": 8,    # Maximum number of regimes for clustering
    "cluster_method": "ward",  # Linkage method for agglomerative clustering

    # Output
    "algo_name": "sigmmd_regime",
    "random_seed": 42,
}

np.random.seed(CONFIG["random_seed"])

# ==============================================================================
# SECTION 1: PATH TRANSFORMATIONS (Section 3.1)
# ==============================================================================

def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute log returns from price series.
    r_t = log(p_t / p_{t-1})

    Args:
        prices: (T,) array of close prices

    Returns:
        (T-1,) array of log returns
    """
    return np.diff(np.log(prices))


def apply_path_transformations(
    returns: np.ndarray,
    lamb: float = np.sqrt(252)
) -> np.ndarray:
    """
    Apply path transformations Φ = φ_time ∘ φ_norm ∘ φ_scale^λ  (Section 3.1)

    1. φ_scale^λ: Scale returns by λ = dt^(-1/2) to annualize
    2. φ_norm: Cumulative sum to convert returns into a path (lead-lag embedding)
    3. φ_time: Time augmentation — add a linear time channel

    Args:
        returns: (T,) array of log returns
        lamb: Scaling factor (default sqrt(252) for daily)

    Returns:
        (T+1, 2) array — time-augmented path [price_channel, time_channel]
    """
    scaled = returns * lamb                               # φ_scale
    path = np.cumsum(np.insert(scaled, 0, 0))             # φ_norm: cumulative sum
    T = len(path)
    time_channel = np.linspace(0, 1, T)                   # φ_time: [0, 1] normalized
    return np.column_stack([path, time_channel])


# ==============================================================================
# SECTION 2: PATH SIGNATURE COMPUTATION (Definition 2.1, Appendix A)
# ==============================================================================

def compute_truncated_signature(
    path: np.ndarray,
    level: int = 2
) -> np.ndarray:
    """
    Compute truncated path signature up to given level.

    For a path X: {0,...,N} → ℝ^d with increments ΔX_k = X_k - X_{k-1}:

    Level 0: scalar 1
    Level 1: S^1_i = Σ_k ΔX_k^i   (d terms — total increment)
    Level 2: S^2_{ij} = Σ_{0<k<l≤N} ΔX_k^i · ΔX_l^j   (d² terms — area)

    The full signature feature vector concatenates all levels:
    Sig_N(X) = [1, S^1_1, ..., S^1_d, S^2_11, S^2_12, ..., S^2_dd]

    Args:
        path: (N, d) array, path values at discrete time points
        level: Truncation level (1 or 2)

    Returns:
        (dim,) array — signature feature vector
        dim = 1 + d + d² + ... + d^level
    """
    N, d = path.shape

    if N < 2:
        raise ValueError(f"Path must have at least 2 points, got {N}")

    increments = np.diff(path, axis=0)  # (N-1, d)

    # Prepend with 1 for level 0
    features = [1.0]

    if level >= 1:
        # Level 1: total increment (d terms, Eq. 5 level-1 term)
        s1 = np.sum(increments, axis=0)  # (d,)
        features.extend(s1.tolist())

    if level >= 2:
        # Level 2: area terms (d² terms)
        # S^2_{ij} = Σ_{0<k<l≤N} ΔX_k^i · ΔX_l^j
        # Efficient: S^2 = Σ_k (cumulative sum up to k · ΔX_{k+1})
        s2 = np.zeros((d, d))
        cumsum = np.zeros(d)
        for k in range(N - 1):
            inc = increments[k]
            s2 += np.outer(cumsum, inc)
            cumsum += inc
        features.extend(s2.flatten().tolist())

    if level >= 3:
        # Level 3: triple iterated integrals (d³ terms)
        # S^3_{ijk} = Σ_{a<b<c} ΔX_a^i · ΔX_b^j · ΔX_c^k
        s3 = np.zeros((d, d, d))
        cumsum2 = np.zeros((d, d))
        cumsum = np.zeros(d)
        for k in range(N - 1):
            inc = increments[k]
            s3 += np.einsum('ij,k->ijk', cumsum2, inc)
            cumsum2 += np.outer(cumsum, inc)
            cumsum += inc
        features.extend(s3.flatten().tolist())

    return np.array(features)


def compute_rbf_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Compute RBF kernel matrix between two sets of signature feature vectors.

    k(x, y) = exp(-||x - y||² / (2σ²))

    Args:
        X: (n, dim) array of signature features
        Y: (m, dim) array of signature features
        sigma: RBF bandwidth

    Returns:
        (n, m) kernel matrix K_{ij} = k(X_i, Y_j)
    """
    # Pairwise squared Euclidean distances
    X_norm = np.sum(X ** 2, axis=1)[:, np.newaxis]  # (n, 1)
    Y_norm = np.sum(Y ** 2, axis=1)[np.newaxis, :]  # (1, m)
    dists_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    dists_sq = np.maximum(dists_sq, 0)  # Numerical stability

    return np.exp(-dists_sq / (2 * sigma ** 2))


# ==============================================================================
# SECTION 3: MMD COMPUTATION (Equation 16)
# ==============================================================================

def mmd_squared(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float = 1.0
) -> float:
    """
    Unbiased estimator of squared MMD between two samples.
    (Equation 16 in the paper)

    D_u^κ(ℙ,ℚ)² = 1/(n(n-1)) Σ_{i≠j} κ(x_i, x_j)
                 - 2/(mn) Σ_i Σ_j κ(x_i, y_j)
                 + 1/(m(m-1)) Σ_{i≠j} κ(y_i, y_j)

    Args:
        X: (n, dim) sample from ℙ
        Y: (m, dim) sample from ℚ
        sigma: RBF bandwidth

    Returns:
        float — squared MMD value (can be slightly negative due to unbiasedness)
    """
    n, m = len(X), len(Y)

    if n < 2 or m < 2:
        return 0.0

    # Within-X kernel terms (excluding diagonal)
    K_XX = compute_rbf_kernel(X, X, sigma)
    np.fill_diagonal(K_XX, 0)
    xx_term = K_XX.sum() / (n * (n - 1))

    # Cross kernel terms
    K_XY = compute_rbf_kernel(X, Y, sigma)
    xy_term = K_XY.sum() / (m * n)

    # Within-Y kernel terms (excluding diagonal)
    K_YY = compute_rbf_kernel(Y, Y, sigma)
    np.fill_diagonal(K_YY, 0)
    yy_term = K_YY.sum() / (m * (m - 1))

    return xx_term - 2 * xy_term + yy_term


# ==============================================================================
# SECTION 4: SUB-PATH EXTRACTION AND ENSEMBLE FORMATION (Definition 3.1)
# ==============================================================================

def extract_sub_paths(
    path: np.ndarray,
    h1: int
) -> np.ndarray:
    """
    Extract overlapping sub-paths of length h1 from the full path.

    Args:
        path: (T, d) full path array
        h1: Sub-path length

    Returns:
        (N1, h1, d) array of sub-paths, N1 = T - h1 + 1
    """
    T = len(path)
    N1 = T - h1 + 1
    if N1 <= 0:
        raise ValueError(f"Path too short ({T}) for sub-path length {h1}")

    # Use stride_tricks for efficient extraction
    shape = (N1, h1, path.shape[1])
    strides = (path.strides[0],) + path.strides
    return np.lib.stride_tricks.as_strided(path, shape=shape, strides=strides).copy()


def form_ensembles(
    sub_paths: np.ndarray,
    h2: int
) -> np.ndarray:
    """
    Form ensembles: groups of h2 consecutive sub-paths.

    Args:
        sub_paths: (N1, h1, d) array of sub-paths
        h2: Ensemble size (number of sub-paths per ensemble)

    Returns:
        (N2, h2, h1, d) array of ensembles, N2 = N1 - h2 + 1
        Ensemble e_i = {s_i, s_{i+1}, ..., s_{i+h2-1}}
    """
    N1 = len(sub_paths)
    N2 = N1 - h2 + 1
    if N2 <= 0:
        raise ValueError(f"Too few sub-paths ({N1}) for ensemble size {h2}")

    shape = (N2, h2) + sub_paths.shape[1:]
    strides = (sub_paths.strides[0],) + sub_paths.strides
    return np.lib.stride_tricks.as_strided(sub_paths, shape=shape, strides=strides).copy()


def compute_ensemble_signatures(
    ensembles: np.ndarray,
    sig_level: int = 2
) -> np.ndarray:
    """
    Compute signature features for all sub-paths, organized by ensemble.

    Args:
        ensembles: (N2, h2, h1, d) array of ensembles
        sig_level: Signature truncation level

    Returns:
        (N2, h2, sig_dim) array — each ensemble is h2 signature vectors
    """
    N2, h2, h1, d = ensembles.shape

    # Reshape to (N2*h2, h1, d) to compute all signatures at once
    flat_paths = ensembles.reshape(-1, h1, d)

    sigs = []
    for i in range(len(flat_paths)):
        sigs.append(compute_truncated_signature(flat_paths[i], sig_level))

    sig_dim = len(sigs[0])
    sig_array = np.array(sigs)

    return sig_array.reshape(N2, h2, sig_dim)


# ==============================================================================
# SECTION 5: AUTO-EVALUATOR (Definition 3.9)
# ==============================================================================

def auto_evaluator(
    ensemble_sigs: np.ndarray,
    L: list,
    L_weights: list,
    prior_window: int,
    alpha: float,
    rbf_sigma: float = 1.0
) -> tuple:
    """
    Non-parametric online regime detection using the auto-evaluator.
    (Definition 3.9, Section 3.2.2)

    For each ensemble i, compute the L-lag auto-evaluation score:
      A_L_i = Σ_{l∈L} w_l · MMD(e_{i-l}, e_i)    (Equation 29)

    Compare against empirical null distribution to detect regime changes.

    Args:
        ensemble_sigs: (N2, h2, sig_dim) signature features per ensemble
        L: List of lag values (e.g., [4, 8, 12])
        L_weights: Weights for each lag value
        prior_window: Number of initial scores for null distribution
        alpha: Confidence level (e.g., 0.95)
        rbf_sigma: RBF kernel bandwidth

    Returns:
        scores: (N2,) array of auto-evaluation scores
        thresholds: (N2,) array of critical thresholds
        change_points: (N2,) boolean array (True = regime change detected)
    """
    N2 = ensemble_sigs.shape[0]
    max_lag = max(L)

    scores = np.full(N2, np.nan)
    thresholds = np.full(N2, np.nan)
    change_points = np.zeros(N2, dtype=bool)

    # Rolling null distribution
    null_dist = []

    for i in range(max_lag, N2):
        # Compute L-lag auto evaluation score (Eq. 29)
        score = 0.0
        valid_lags = 0
        for l_idx, l in enumerate(L):
            if i - l >= 0:
                # Compare current ensemble e_i against lagged ensemble e_{i-l}
                e_current = ensemble_sigs[i]     # (h2, sig_dim)
                e_lagged = ensemble_sigs[i - l]  # (h2, sig_dim)
                mmd_val = mmd_squared(e_current, e_lagged, rbf_sigma)
                score += L_weights[l_idx] * mmd_val
                valid_lags += 1

        if valid_lags > 0:
            score /= sum(w for idx, w in enumerate(L_weights) if i - L[idx] >= 0)
        scores[i] = score

        # Determine critical threshold from null distribution
        if len(null_dist) >= max(20, prior_window // 4):
            # Enough data — compute empirical quantile
            threshold = np.percentile(null_dist, alpha * 100)
            thresholds[i] = threshold

            if score > threshold:
                change_points[i] = True
                # Don't add change-point scores to null distribution
            else:
                null_dist.append(score)
        else:
            thresholds[i] = np.nan
            # Build null distribution
            null_dist.append(score)

        # Keep rolling window of prior_window most recent scores
        if len(null_dist) > prior_window:
            null_dist = null_dist[-prior_window:]

    return scores, thresholds, change_points


# ==============================================================================
# SECTION 6: REGIME CLUSTERING (Section 3.3, Section 5)
# ==============================================================================

def cluster_ensembles(
    ensemble_sigs: np.ndarray,
    n_clusters: int,
    rbf_sigma: float = 1.0,
    method: str = "average"
) -> np.ndarray:
    """
    Agglomerative hierarchical clustering of ensembles into regimes.
    (Section 3.3 — path-wise regime clustering)

    Uses the MMD-induced distance matrix between ensembles.
    Self-contained implementation — no external clustering libraries.

    Args:
        ensemble_sigs: (N2, h2, sig_dim) signature features
        n_clusters: Number of regime clusters
        rbf_sigma: RBF kernel bandwidth
        method: Linkage method ('average', 'single', 'complete')

    Returns:
        (N2,) array of cluster labels (0-indexed)
    """
    N2 = ensemble_sigs.shape[0]
    if n_clusters >= N2:
        return np.arange(N2, dtype=int)

    # Compute mean signature per ensemble (barycenter in feature space)
    mean_sigs = ensemble_sigs.mean(axis=1)  # (N2, sig_dim)

    # Compute pairwise kernel distance matrix
    # d(x,y)² = k(x,x) + k(y,y) - 2k(x,y)
    K = compute_rbf_kernel(mean_sigs, mean_sigs, rbf_sigma)  # (N2, N2)
    diag = np.diag(K)
    D_sq = diag[:, np.newaxis] + diag[np.newaxis, :] - 2 * K
    D_sq = np.maximum(D_sq, 0)
    D = np.sqrt(D_sq)

    # Agglomerative clustering
    labels = agglomerative_clustering(D, n_clusters, method)
    return labels


def agglomerative_clustering(
    D: np.ndarray,
    n_clusters: int,
    method: str = "average"
) -> np.ndarray:
    """
    Simple agglomerative hierarchical clustering without scipy.
    Implements average-linkage clustering.

    Args:
        D: (N, N) pairwise distance matrix (symmetric, zero diagonal)
        n_clusters: Target number of clusters
        method: Linkage method ('average', 'single', 'complete')

    Returns:
        (N,) array of cluster labels (0-indexed)
    """
    N = len(D)
    # Each point starts in its own cluster
    labels = np.arange(N)
    n_current = N

    # Precompute cluster sizes
    sizes = np.ones(N)

    while n_current > n_clusters:
        # Find the two closest clusters (min distance between different clusters)
        min_dist = np.inf
        merge_i, merge_j = -1, -1

        # Get unique cluster IDs
        unique_ids = np.unique(labels)
        n_u = len(unique_ids)

        for a in range(n_u):
            for b in range(a + 1, n_u):
                ci, cj = unique_ids[a], unique_ids[b]
                mask_i = labels == ci
                mask_j = labels == cj

                # Distance between clusters based on linkage method
                dists = D[np.ix_(mask_i, mask_j)]
                if method == "single":
                    d = dists.min()
                elif method == "complete":
                    d = dists.max()
                else:  # average
                    d = dists.mean()

                if d < min_dist:
                    min_dist = d
                    merge_i, merge_j = ci, cj

        if merge_i < 0 or merge_j < 0:
            break

        # Merge: assign all points in cluster j to cluster i
        labels[labels == merge_j] = merge_i
        n_current -= 1

    # Renumber clusters to 0..k-1
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(unique)}
    return np.array([mapping[l] for l in labels])


def resolve_regime_labels(
    regimes: np.ndarray,
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    h1: int
) -> dict:
    """
    Assign human-readable labels to regimes based on return/vol characteristics.
    Labels: 'bull' (positive return, low vol), 'bear' (negative return, high vol),
            'volatile' (high vol), 'quiet' (low vol), 'normal_<N>'

    Args:
        regimes: (N2,) regime labels per ensemble
        returns: (T,) daily log returns
        dates: (T,) date index
        h1: sub-path length

    Returns:
        dict mapping regime_id → label string
    """
    unique_regimes = np.unique(regimes)
    regime_returns = {}
    regime_vols = {}

    # Map ensembles back to dates (each ensemble covers h1*h2 trading days)
    for rid in unique_regimes:
        mask = regimes == rid
        n_ensembles = mask.sum()
        # Approximate: ensemble i covers dates[i : i + h1 * h2]
        idx_list = []
        for i in np.where(mask)[0]:
            idx_list.extend(range(i, min(i + h1, len(returns))))
        if idx_list:
            r = returns[idx_list]
            regime_returns[rid] = float(np.mean(r))
            regime_vols[rid] = float(np.std(r))

    all_vols = list(regime_vols.values())
    vol_median = float(np.median(all_vols))
    vol_p25 = float(np.percentile(all_vols, 25))
    vol_p75 = float(np.percentile(all_vols, 75))

    labels_map = {}
    for rid in unique_regimes:
        mu = regime_returns.get(rid, 0)
        sigma = regime_vols.get(rid, 0)
        if mu > 0 and sigma < vol_median:
            labels_map[str(rid)] = "bull"
        elif mu < 0 and sigma > vol_median:
            labels_map[str(rid)] = "bear"
        elif sigma > vol_p75:
            labels_map[str(rid)] = "high_volatility"
        elif sigma < vol_p25:
            labels_map[str(rid)] = "low_volatility"
        else:
            labels_map[str(rid)] = f"normal_{int(rid)}"

    return labels_map


# ==============================================================================
# SECTION 7: OUTPUT GENERATION (per output_spec.json v1.0)
# ==============================================================================

def generate_output_files(
    dates: pd.DatetimeIndex,
    closing_prices: np.ndarray,
    scores: np.ndarray,
    change_points: np.ndarray,
    ensemble_regimes: np.ndarray,
    regime_labels_map: dict,
    config: dict,
    output_dir: Path
):
    """
    Generate output files conforming to output_spec.json v1.0:
    - {algo_name}_regimes.csv
    - {algo_name}_metadata.json

    Maps ensemble-level regime labels back to daily dates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    algo_name = config["algo_name"]
    h1, h2 = config["h1"], config["h2"]

    # Map ensemble-level regimes to daily dates
    # Ensemble i covers approximate date range: dates[i] to dates[i + h1*h2 - 1]
    n_dates = len(dates)
    daily_regimes = np.full(n_dates, -1, dtype=int)
    daily_confidence = np.zeros(n_dates)
    daily_transition = np.zeros(n_dates, dtype=int)
    daily_scores = np.full(n_dates, np.nan)

    N2 = len(ensemble_regimes)

    for i in range(N2):
        start_idx = i
        end_idx = min(i + h1 * h2, n_dates)
        for j in range(start_idx, end_idx):
            daily_regimes[j] = ensemble_regimes[i]
            # Confidence: inversely proportional to normalized score
            if not np.isnan(scores[i]) and i < N2 - 1:
                # Higher confidence when score is low (stable regime)
                score_max = np.nanmax(scores)
                if score_max > 0:
                    daily_confidence[j] = max(0.0, min(1.0, 1.0 - scores[i] / score_max))
                else:
                    daily_confidence[j] = 1.0
            daily_scores[j] = scores[i]

    # Mark transitions
    for j in range(1, n_dates):
        if daily_regimes[j] != daily_regimes[j - 1] and daily_regimes[j] >= 0:
            daily_transition[j] = 1

    # Filter to evaluation window
    eval_start = pd.Timestamp(config["eval_start"])
    eval_end = pd.Timestamp(config["eval_end"])
    mask = (dates >= eval_start) & (dates <= eval_end)
    eval_dates = dates[mask]
    eval_regimes = daily_regimes[mask]
    eval_confidence = daily_confidence[mask]
    eval_transition = daily_transition[mask]
    eval_scores = daily_scores[mask]

    # Build regime labels column
    regime_label_col = []
    for rid in eval_regimes:
        if rid >= 0 and str(rid) in regime_labels_map:
            regime_label_col.append(regime_labels_map[str(rid)])
        else:
            regime_label_col.append("")

    # Write CSV
    df = pd.DataFrame({
        "date": eval_dates.strftime("%Y-%m-%d"),
        "predicted_regime": eval_regimes,
        "confidence": eval_confidence.round(4),
        "transition_flag": eval_transition,
        "regime_label": regime_label_col,
    })

    # Ensure no NaN in required columns
    df["predicted_regime"] = df["predicted_regime"].fillna(0).astype(int)
    df["confidence"] = df["confidence"].fillna(0.5).clip(0, 1)
    df["transition_flag"] = df["transition_flag"].fillna(0).astype(int)

    csv_path = output_dir / f"{algo_name}_regimes.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path} ({len(df)} rows)")

    # Write metadata JSON
    num_regimes = int(df["predicted_regime"].max()) + 1
    metadata = {
        "algorithm": algo_name,
        "algorithm_full_name": "Signature MMD-based Regime Detection (Auto-Evaluator)",
        "algorithm_family": "changepoint",
        "paper_reference": "arXiv:2306.15835",
        "num_regimes": num_regimes,
        "regime_labels_map": regime_labels_map,
        "parameters": {
            "h1": config["h1"],
            "h2": config["h2"],
            "sig_level": config["sig_level"],
            "rbf_sigma": config["rbf_sigma"],
            "alpha": config["alpha"],
            "prior_window": config["prior_window"],
            "L": config["L"],
            "L_weights": config["L_weights"],
            "annualize_scale": config["annualize_scale"],
            "cluster_method": config["cluster_method"],
            "training_window_days": None,
            "features_used": [
                "log_return",
                "daily_vol_21d",
                "cumulative_return_21d"
            ],
            "random_seed": config["random_seed"],
        },
        "training_window_days": None,
        "features_used": [
            "log_return",
            "daily_vol_21d",
            "cumulative_return_21d"
        ],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_spec_version": "1.0",
    }

    meta_path = output_dir / f"{algo_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {meta_path}")

    return df, metadata


# ==============================================================================
# SECTION 8: MAIN PIPELINE
# ==============================================================================

def main():
    """Run the full regime detection pipeline."""

    print("=" * 70)
    print("REGIME DETECTION via SIGNATURE MMD (Auto-Evaluator)")
    print("Paper: Horvath & Issa 2023, arXiv:2306.15835")
    print("=" * 70)
    print(f"Config: h1={CONFIG['h1']}, h2={CONFIG['h2']}, sig_level={CONFIG['sig_level']}")
    print(f"MMD: {CONFIG['kernel']} kernel, σ={CONFIG['rbf_sigma']}")
    print(f"Auto-evaluator: L={CONFIG['L']}, α={CONFIG['alpha']}, prior={CONFIG['prior_window']}")
    print()

    # --- Load data ---
    print("1. Loading S&P 500 data...")
    # Use the clean CSV from T1 (simple header: date,open,high,low,close,adj_close,volume)
    data_path = Path("/opt/data/kanban/workspaces/t_f9e0a695/sp500_data/sp500_ohlcv_20yr.csv")
    if not data_path.exists():
        # Fallback to local copy
        data_path = Path(__file__).parent / "sp500_ohlcv_20yr.csv"

    df = pd.read_csv(data_path, parse_dates=["date"], index_col="date")
    close = df["close"].values  # lowercase from the clean CSV
    dates = df.index
    n_dates = len(dates)
    print(f"   Loaded {n_dates} trading days: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print()

    # --- Path transformations ---
    print("2. Computing log returns and path transformations...")
    log_rets = compute_log_returns(close)
    path = apply_path_transformations(log_rets, CONFIG["annualize_scale"])
    print(f"   Path shape: {path.shape} (T={len(path)}, d={path.shape[1]})")
    print()

    # --- Extract sub-paths and form ensembles ---
    print("3. Extracting sub-paths and forming ensembles...")
    sub_paths = extract_sub_paths(path, CONFIG["h1"])
    N1 = len(sub_paths)
    print(f"   Sub-paths: {N1} (length h1={CONFIG['h1']})")

    ensembles = form_ensembles(sub_paths, CONFIG["h2"])
    N2 = len(ensembles)
    print(f"   Ensembles:  {N2} (size h2={CONFIG['h2']})")
    print()

    # --- Compute signature features ---
    print("4. Computing truncated path signatures...")
    ensemble_sigs = compute_ensemble_signatures(ensembles, CONFIG["sig_level"])
    sig_dim = ensemble_sigs.shape[-1]
    print(f"   Signature dimension: {sig_dim} (level={CONFIG['sig_level']}, d={path.shape[1]})")
    print()

    # --- Auto-evaluator: online regime detection ---
    print("5. Running auto-evaluator for regime change detection...")
    scores, thresholds, change_points = auto_evaluator(
        ensemble_sigs,
        CONFIG["L"],
        CONFIG["L_weights"],
        CONFIG["prior_window"],
        CONFIG["alpha"],
        CONFIG["rbf_sigma"]
    )

    n_changes = change_points.sum()
    n_valid = np.sum(~np.isnan(scores))
    print(f"   Valid scores: {n_valid}/{N2}")
    print(f"   Regime changes detected: {n_changes}")
    print(f"   Score statistics: mean={np.nanmean(scores):.6f}, std={np.nanstd(scores):.6f}")
    print(f"   Score range: [{np.nanmin(scores):.6f}, {np.nanmax(scores):.6f}]")
    print()

    # --- Regime clustering ---
    print("6. Clustering regimes via agglomerative clustering...")
    # Determine number of clusters based on change point density
    n_clusters = max(2, min(CONFIG["n_regimes_max"], n_changes + 2))
    ensemble_regimes = cluster_ensembles(
        ensemble_sigs,
        n_clusters,
        CONFIG["rbf_sigma"],
        CONFIG["cluster_method"]
    )
    print(f"   Clustered into {n_clusters} regimes")
    for rid in range(n_clusters):
        count = (ensemble_regimes == rid).sum()
        print(f"     Regime {rid}: {count} ensembles ({count/N2*100:.1f}%)")
    print()

    # --- Label regimes ---
    print("7. Assigning regime labels...")
    regime_labels_map = resolve_regime_labels(
        ensemble_regimes,
        log_rets,
        dates,
        CONFIG["h1"]
    )
    for rid, label in regime_labels_map.items():
        count = (ensemble_regimes == int(rid)).sum()
        print(f"     Regime {rid} → '{label}' ({count} ensembles)")
    print()

    # --- Generate output files ---
    print("8. Generating output files...")
    output_dir = Path("/opt/data/kanban/workspaces/t_b22593ba/paper-3947905") / \
                 "evaluation-plan" / "outputs" / CONFIG["algo_name"]
    df_out, metadata = generate_output_files(
        dates,
        close,
        scores,
        change_points,
        ensemble_regimes,
        regime_labels_map,
        CONFIG,
        output_dir
    )

    # --- Summary ---
    print()
    print("=" * 70)
    print("REGIME DETECTION SUMMARY")
    print("=" * 70)
    print(f"Algorithm:     {metadata['algorithm_full_name']}")
    print(f"Paper:         {metadata['paper_reference']}")
    print(f"Num regimes:   {metadata['num_regimes']}")
    print(f"Regime labels: {metadata['regime_labels_map']}")
    print(f"Regime changes detected: {n_changes}")
    csv_file = output_dir / f"{CONFIG['algo_name']}_regimes.csv"
    json_file = output_dir / f"{CONFIG['algo_name']}_metadata.json"
    print(f"Output CSV:    {csv_file}")
    print(f"Output JSON:   {json_file}")

    # Print key transition dates
    change_dates = []
    for i in range(N2):
        if change_points[i] and i < len(dates):
            change_dates.append(dates[i].strftime("%Y-%m-%d"))

    if change_dates:
        print(f"\nDetected regime change dates (top 20):")
        for d in change_dates[:20]:
            print(f"  {d}")
        if len(change_dates) > 20:
            print(f"  ... and {len(change_dates) - 20} more")
    print("=" * 70)

    return df_out, metadata, scores, change_points


if __name__ == "__main__":
    df_out, metadata, scores, change_points = main()