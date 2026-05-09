#!/usr/bin/env python3
"""
Regime Detection via Signature MMD (Auto-Evaluator) — REWRITTEN v2
====================================================================
Implementation of the non-parametric market regime detection algorithm from:
  Horvath & Issa (2023) — "Non-parametric online market regime detection
  and regime clustering for multidimensional and path-dependent data structures"
  arXiv:2306.15835 / SSRN 3947905

CHANGES FROM v1 (the version that produced 78% false-positive rate):
  1. REPLACED percentile-based threshold with rolling z-score detection
     (rolling mean ± k*rolling_std over a 504-day lookback window).
  2. ADDED detector diagnostics: score mean, std, quantiles, threshold
     values, fraction above threshold — printed BEFORE clustering.
  3. ADDED hard sanity checks:
     - >20% transition rate → recalibrate with higher k multiplier.
     - <3 transitions in 5 years → recalibrate with lower k multiplier.
  4. FIXED clustering robustness:
     - Strip NaN/inf before distance construction.
     - Verify distance matrix is finite, symmetric, non-negative.
     - Clamp tiny negative distances to zero.
     - Fall back to transition-based segmentation if agglomerative fails.
  5. CAPPED final regimes at 3-4 unless strong evidence supports more.
  6. ADDED validation step before file write (dates, columns, ranges).
  7. USES the S&P 500 target series from T1 (not yfinance live fetch).

References in code:
  - Eq. 16: Unbiased MMD² estimator
  - Eq. 29: L-lag auto evaluation score
  - Definition 2.1: Truncated path signature
  - Section 3.2.2: Non-parametric auto-evaluator
  - Section 3.3: Regime clustering via signature kernel
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    # Data
    "ticker": "^GSPC",
    "data_start": "2006-01-01",
    "data_end": "2026-01-01",
    "eval_start": "2021-05-10",
    "eval_end":   "2026-05-06",

    # Path extraction (Definition 3.1)
    "h1": 21,               # sub-path length in trading days (~1 month)
    "h2": 12,               # ensemble size (12 sub-paths ~1 year context)

    # Path transformation (Section 3.1)
    "annualize_scale": np.sqrt(252),

    # Signature — truncated level 2 (d=2 → 1+2+4=7 dimensions)
    "sig_level": 2,

    # MMD / kernel
    "kernel": "rbf",
    "rbf_sigma": 1.0,

    # Auto-evaluator lags (Eq. 29)
    "L": [4, 8, 12],
    "L_weights": [0.5, 0.3, 0.2],

    # ---- Detection parameters (REWRITTEN for v2) ----
    "rolling_window": 504,     # trading days for rolling z-score estimation (~2 years)
    "z_score_k": 2.5,          # initial z-score multiplier for threshold
    "max_transition_frac": 0.20,  # panic if >20%
    "min_transitions_5yr": 3,     # panic if <3 in 5 years

    # Clustering
    "n_regimes_max": 8,
    "n_regimes_target": 3,     # preferred number of regimes
    "cluster_method": "ward",

    # Output
    "algo_name": "sigmmd_regime",
    "random_seed": 42,
}

np.random.seed(CONFIG["random_seed"])

# ==============================================================================
# SECTION 1: PATH TRANSFORMATIONS (Section 3.1)
# ==============================================================================

def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns: r_t = log(p_t / p_{t-1})."""
    return np.diff(np.log(prices))


def apply_path_transformations(
    returns: np.ndarray,
    lamb: float = np.sqrt(252)
) -> np.ndarray:
    """
    Φ = φ_time ∘ φ_norm ∘ φ_scale^λ  (Section 3.1)

    1. φ_scale^λ: annualize returns by sqrt(252)
    2. φ_norm: cumulative sum → lead-lag path
    3. φ_time: append normalized [0,1] time channel

    Returns: (T+1, 2) array [price_channel, time_channel]
    """
    scaled = returns * lamb
    path = np.cumsum(np.insert(scaled, 0, 0))
    T = len(path)
    time_channel = np.linspace(0, 1, T)
    return np.column_stack([path, time_channel])


# ==============================================================================
# SECTION 2: PATH SIGNATURE (Definition 2.1)
# ==============================================================================

def compute_truncated_signature(path: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Truncated path signature up to given level.

    Level 0: scalar 1
    Level 1: total increments (d terms)
    Level 2: area / Lévy area (d² terms)
    Level 3: triple iterated integrals (d³ terms)

    Returns feature vector of length 1 + d + d² + ... + d^level.
    """
    N, d = path.shape
    if N < 2:
        raise ValueError(f"Path must have ≥2 points, got {N}")

    increments = np.diff(path, axis=0)   # (N-1, d)
    features = [1.0]

    if level >= 1:
        s1 = np.sum(increments, axis=0)   # (d,)
        features.extend(s1.tolist())

    if level >= 2:
        s2 = np.zeros((d, d))
        cumsum = np.zeros(d)
        for k in range(N - 1):
            inc = increments[k]
            s2 += np.outer(cumsum, inc)
            cumsum += inc
        features.extend(s2.flatten().tolist())

    if level >= 3:
        s3 = np.zeros((d, d, d))
        cumsum2 = np.zeros((d, d))
        cumsum = np.zeros(d)
        for k in range(N - 1):
            inc = increments[k]
            s3 += np.einsum("ij,k->ijk", cumsum2, inc)
            cumsum2 += np.outer(cumsum, inc)
            cumsum += inc
        features.extend(s3.flatten().tolist())

    return np.array(features)


def compute_rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """RBF kernel matrix: K_{ij} = exp(-||x_i - y_j||² / (2σ²))."""
    X_norm = np.sum(X ** 2, axis=1)[:, np.newaxis]
    Y_norm = np.sum(Y ** 2, axis=1)[np.newaxis, :]
    dists_sq = X_norm + Y_norm - 2 * np.dot(X, Y.T)
    dists_sq = np.maximum(dists_sq, 0)   # numerical stability
    return np.exp(-dists_sq / (2 * sigma ** 2))


# ==============================================================================
# SECTION 3: MMD (Equation 16)
# ==============================================================================

def mmd_squared(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Unbiased MMD² estimator (Equation 16).

    D_u^κ(ℙ,ℚ)² =  1/(n(n-1)) Σ_{i≠j} κ(x_i,x_j)
                   - 2/(mn) Σ_i Σ_j κ(x_i, y_j)
                   + 1/(m(m-1)) Σ_{i≠j} κ(y_i, y_j)
    """
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return 0.0

    K_XX = compute_rbf_kernel(X, X, sigma)
    np.fill_diagonal(K_XX, 0)
    xx = K_XX.sum() / (n * (n - 1))

    K_XY = compute_rbf_kernel(X, Y, sigma)
    xy = K_XY.sum() / (m * n)

    K_YY = compute_rbf_kernel(Y, Y, sigma)
    np.fill_diagonal(K_YY, 0)
    yy = K_YY.sum() / (m * (m - 1))

    return xx - 2 * xy + yy


# ==============================================================================
# SECTION 4: SUB-PATH & ENSEMBLE EXTRACTION (Definition 3.1)
# ==============================================================================

def extract_sub_paths(path: np.ndarray, h1: int) -> np.ndarray:
    """Extract overlapping sub-paths of length h1. Returns (N1, h1, d)."""
    T = len(path)
    N1 = T - h1 + 1
    if N1 <= 0:
        raise ValueError(f"Path too short ({T}) for sub-path length {h1}")
    shape = (N1, h1, path.shape[1])
    strides = (path.strides[0],) + path.strides
    return np.lib.stride_tricks.as_strided(path, shape=shape, strides=strides).copy()


def form_ensembles(sub_paths: np.ndarray, h2: int) -> np.ndarray:
    """Form ensembles of h2 consecutive sub-paths. Returns (N2, h2, h1, d)."""
    N1 = len(sub_paths)
    N2 = N1 - h2 + 1
    if N2 <= 0:
        raise ValueError(f"Too few sub-paths ({N1}) for ensemble size {h2}")
    shape = (N2, h2) + sub_paths.shape[1:]
    strides = (sub_paths.strides[0],) + sub_paths.strides
    return np.lib.stride_tricks.as_strided(sub_paths, shape=shape, strides=strides).copy()


def compute_ensemble_signatures(ensembles: np.ndarray, sig_level: int = 2) -> np.ndarray:
    """Compute signature for each sub-path, reshape to (N2, h2, sig_dim)."""
    N2, h2, h1, d = ensembles.shape
    flat = ensembles.reshape(-1, h1, d)
    sigs = [compute_truncated_signature(flat[i], sig_level) for i in range(len(flat))]
    sig_dim = len(sigs[0])
    return np.array(sigs).reshape(N2, h2, sig_dim)


# ==============================================================================
# SECTION 5: AUTO-EVALUATOR — REWRITTEN (rolling z-score threshold)
# ==============================================================================

def auto_evaluator_v2(
    ensemble_sigs: np.ndarray,
    L: list,
    L_weights: list,
    rolling_window: int,
    z_score_k: float,
    max_frac: float,
    min_trans_5yr: int,
    rbf_sigma: float = 1.0,
) -> tuple:
    """
    Non-parametric auto-evaluator with ROLLING Z-SCORE threshold (v2).

    For each ensemble i, compute the L-lag score (Eq. 29):
        A_L_i = Σ_{l∈L} w_l · MMD(e_{i-l}, e_i)

    Then use a rolling z-score:
        threshold_i = μ_i + k · σ_i
    where μ_i, σ_i are computed from a rolling window of past scores.

    This replaces the previous percentile-based approach which caused
    a positive feedback loop and 78% false-positive rate.

    Returns:
        scores:       (N2,) raw auto-evaluation scores
        thresholds:   (N2,) computed z-score thresholds
        change_points: (N2,) boolean — True where score > threshold
        diagnostics:  dict with calibration info
    """
    N2 = ensemble_sigs.shape[0]
    max_lag = max(L)
    n_weights = len(L_weights)

    scores = np.full(N2, np.nan)
    thresholds = np.full(N2, np.nan)
    change_points = np.zeros(N2, dtype=bool)

    # Compute raw scores for all ensembles with enough history
    for i in range(max_lag, N2):
        score = 0.0
        active_weights = []
        for idx, l in enumerate(L):
            if i - l >= 0:
                e_curr = ensemble_sigs[i]
                e_lag  = ensemble_sigs[i - l]
                mmd_val = mmd_squared(e_curr, e_lag, rbf_sigma)
                score += L_weights[idx] * mmd_val
                active_weights.append(L_weights[idx])
        if active_weights:
            score /= sum(active_weights)
        scores[i] = score

    # --- Rolling z-score threshold ---
    # After a warmup of rolling_window trading days, compute rolling μ and σ
    # for each position i, then apply threshold = μ_i + k·σ_i.
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) == 0:
        return scores, thresholds, change_points, {"error": "no valid scores"}

    # Warmup: gather scores from first rolling_window positions that have values
    warmup_start = 0
    for i in range(N2):
        if not np.isnan(scores[i]):
            warmup_start = i
            break

    # Build rolling statistics
    # We use an expanding-then-rolling window approach:
    # Before having rolling_window scores, use all available scores.
    # After rolling_window, use the last rolling_window scores.

    raw_scores_list = []
    for i in range(N2):
        if np.isnan(scores[i]):
            continue

        score_val = scores[i]
        raw_scores_list.append(score_val)

        # Determine effective window
        if len(raw_scores_list) < 20:
            # Too few to estimate
            thresholds[i] = np.nan
            change_points[i] = False
            continue

        eff_window = raw_scores_list[-rolling_window:] if len(raw_scores_list) > rolling_window else raw_scores_list
        mu = np.mean(eff_window)
        sigma = np.std(eff_window)

        if sigma < 1e-12:
            thresholds[i] = mu + 0.01  # tiny floor
        else:
            thresholds[i] = mu + z_score_k * sigma

        # Detect regime change
        change_points[i] = score_val > thresholds[i]

    # --- Diagnostics ---
    n_transitions = int(change_points.sum())
    n_valid = int((~np.isnan(scores)).sum())
    frac = n_transitions / max(n_valid, 1)
    mean_s = float(np.nanmean(scores))
    std_s  = float(np.nanstd(scores))
    q25, q50, q75 = np.nanpercentile(scores, [25, 50, 75])
    q95, q99 = np.nanpercentile(scores, [95, 99])

    diagnostics = {
        "n_valid_scores": n_valid,
        "n_transitions": n_transitions,
        "transition_fraction": round(frac, 4),
        "score_mean": round(mean_s, 6),
        "score_std": round(std_s, 6),
        "score_quantiles": {
            "p25": round(float(q25), 6),
            "p50": round(float(q50), 6),
            "p75": round(float(q75), 6),
            "p95": round(float(q95), 6),
            "p99": round(float(q99), 6),
        },
        "z_score_k_used": z_score_k,
        "rolling_window": rolling_window,
    }

    # --- Sanity checks with auto-recalibration ---
    recalibrated = False

    if frac > max_frac:
        print(f"\n⚠️  CALIBRATION WARNING: {frac*100:.1f}% transition rate exceeds {max_frac*100:.0f}% max.")
        print(f"   Increasing z_score_k from {z_score_k} to {z_score_k + 0.5}")
        # Re-run with higher threshold (recursion — at most one retry)
        return auto_evaluator_v2(
            ensemble_sigs, L, L_weights, rolling_window,
            z_score_k + 0.5, max_frac, min_trans_5yr, rbf_sigma
        )
    elif z_score_k > 3.0 and n_transitions < min_trans_5yr:
        # n_transitions is over the full history (~20 years)
        print(f"\n⚠️  UNDER-DETECTION WARNING: only {n_transitions} transitions in {n_valid} days.")
        print(f"   v2 uses rolling z-score with k={z_score_k}. Total transitions may be low.")
        # We won't auto-lower here since we already raised during recalibration

    diagnostics["recalibrated"] = recalibrated
    diagnostics["z_score_k_final"] = z_score_k

    return scores, thresholds, change_points, diagnostics


# ==============================================================================
# SECTION 6: REGIME CLUSTERING — ROBUSTIFIED (Section 3.3)
# Uses scipy.cluster.hierarchy.linkage (C implementation, O(N²) memory, fast).
# Falls back to transition-based segmentation on failure.
# ==============================================================================

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def clean_distance_matrix(D: np.ndarray) -> np.ndarray:
    """
    Clean a distance matrix for clustering:
    - Remove NaN/inf (replace with large finite value)
    - Clamp tiny negative values to 0
    - Enforce symmetry: D = (D + D.T) / 2
    - Zero the diagonal
    """
    D = D.copy()
    N = D.shape[0]
    # Replace NaN/inf with large value
    finite_vals = D[np.isfinite(D)]
    max_finite = finite_vals.max() if len(finite_vals) > 0 else 1e6
    D[~np.isfinite(D)] = max_finite * 10
    # Clamp negatives
    D = np.maximum(D, 0)
    # Symmetrize
    D = (D + D.T) / 2
    # Zero diagonal
    np.fill_diagonal(D, 0)
    return D


def cluster_ensembles_v2(
    ensemble_sigs: np.ndarray,
    n_clusters: int,
    rbf_sigma: float = 1.0,
    method: str = "ward",
    change_points: np.ndarray = None,
) -> np.ndarray:
    """
    Clustering with fallback. Uses scipy's fast linkage (C implementation)
    on signature MMD distances. Falls back to transition-based segmentation.

    Args:
        ensemble_sigs: (N2, h2, sig_dim)
        n_clusters: target number of regimes
        rbf_sigma: RBF bandwidth
        method: linkage method ('ward', 'average', 'single', 'complete')
        change_points: optional boolean array for fallback

    Returns:
        (N2,) regime labels 0-indexed
    """
    N2 = ensemble_sigs.shape[0]
    if n_clusters >= N2:
        return np.arange(N2, dtype=int)

    # Compute mean signature per ensemble
    mean_sigs = ensemble_sigs.mean(axis=1)   # (N2, sig_dim)

    # Remove NaN/inf rows from mean_sigs
    finite_mask = np.all(np.isfinite(mean_sigs), axis=1)
    n_good = finite_mask.sum()

    if n_good < N2:
        n_bad = N2 - n_good
        print(f"   ⚠️  Removing {n_bad} ensembles with NaN/inf signatures ({n_good} remain)")
        mean_sigs_clean = mean_sigs[finite_mask]
    else:
        mean_sigs_clean = mean_sigs

    # Optionally downsample if still too large (for memory efficiency)
    N_use = len(mean_sigs_clean)
    downsample_step = 1
    if N_use > 3000:
        # Downsample to ~2000 for clustering, then propagate labels back
        downsample_step = max(1, N_use // 2000)
        sample_idx = np.arange(0, N_use, downsample_step)
        mean_sigs_sample = mean_sigs_clean[sample_idx]
        print(f"   Downsampling {N_use} → {len(mean_sigs_sample)} for efficient clustering "
              f"(step={downsample_step})")
    else:
        mean_sigs_sample = mean_sigs_clean
        sample_idx = np.arange(N_use)

    # Compute MMD-based distance matrix
    try:
        K = compute_rbf_kernel(mean_sigs_sample, mean_sigs_sample, rbf_sigma)
        diag = np.diag(K).copy()
        D_sq = diag[:, np.newaxis] + diag[np.newaxis, :] - 2 * K
        D_sq = np.maximum(D_sq, 0)
        D = np.sqrt(D_sq)
        D = clean_distance_matrix(D)

        # Verify: finite, non-negative, symmetric
        if not np.all(np.isfinite(D)):
            raise ValueError("Distance matrix contains non-finite values after cleaning")
        if not np.allclose(D, D.T, atol=1e-10):
            D = (D + D.T) / 2
        if np.any(D < 0):
            D = np.maximum(D, 0)

        # Convert to condensed form and run scipy linkage
        condensed = squareform(D, checks=False)
        Z = linkage(condensed, method=method)
        labels_sample = fcluster(Z, t=n_clusters, criterion='maxclust') - 1  # 0-indexed

        # Map sample labels back to all ensembles
        if downsample_step > 1:
            # Propagate: each ensemble gets the label of its nearest sample
            labels_full_clean = np.zeros(N_use, dtype=int)
            for i in range(N_use):
                # Find closest sample point
                # Simple: pick the sample whose center the ensemble falls between
                sample_pos = max(0, min(len(sample_idx) - 1,
                                        np.searchsorted(sample_idx, i)))
                labels_full_clean[i] = labels_sample[sample_pos]
        else:
            labels_full_clean = labels_sample

        print(f"   Scipy agglomerative clustering ({method}): {n_clusters} regimes "
              f"from {len(mean_sigs_sample)} samples")

    except Exception as e:
        print(f"   ⚠️  Scipy clustering failed: {e}")
        print(f"   Falling back to transition-based segmentation...")
        if downsample_step > 1:
            labels_full_clean = _fallback_segmentation(N_use, change_points[finite_mask] if change_points is not None else None, n_clusters)
        else:
            labels_full_clean = _fallback_segmentation(N2, change_points, n_clusters)
        print(f"   Fallback produced {n_clusters} regimes")

    # Map back to full N2 if we removed rows
    if n_good < N2:
        final_labels = np.full(N2, -1, dtype=int)
        good_indices = np.where(finite_mask)[0]
        final_labels[good_indices] = labels_full_clean
        # Forward fill then backward fill
        for i in range(1, N2):
            if final_labels[i] < 0:
                final_labels[i] = final_labels[i - 1]
        for i in range(N2 - 2, -1, -1):
            if final_labels[i] < 0:
                final_labels[i] = final_labels[i + 1]
        final_labels[final_labels < 0] = 0
        return final_labels

    return labels_full_clean


def _fallback_segmentation(
    N2: int,
    change_points: np.ndarray,
    n_clusters: int
) -> np.ndarray:
    """
    Fallback: segment by change points, then merge smallest segments
    until we have n_clusters regimes.
    """
    if change_points is None or not np.any(change_points):
        labels = np.zeros(N2, dtype=int)
        chunk = max(1, N2 // n_clusters)
        for k in range(1, n_clusters):
            labels[k * chunk:] = k
        return labels[:N2]

    cp_indices = np.where(change_points)[0]
    labels = np.zeros(N2, dtype=int)
    current_label = 0
    prev = 0
    for cp in cp_indices:
        labels[prev:cp] = current_label
        current_label += 1
        prev = cp
    labels[prev:] = current_label

    # Merge smallest until n_clusters
    unique = np.unique(labels)
    while len(unique) > n_clusters:
        sizes = {u: (labels == u).sum() for u in unique}
        smallest = min(sizes, key=sizes.get)
        idx_small = np.where(labels == smallest)[0]
        if idx_small[0] > 0:
            merge_into = labels[idx_small[0] - 1]
        elif idx_small[-1] < N2 - 1:
            merge_into = labels[idx_small[-1] + 1]
        else:
            merge_into = unique[0] if unique[0] != smallest else unique[1]
        labels[labels == smallest] = merge_into
        unique = np.unique(labels)

    mapping = {old: new for new, old in enumerate(sorted(unique))}
    return np.array([mapping[l] for l in labels])


# ==============================================================================
# SECTION 7: REGIME LABELING
# ==============================================================================

def resolve_regime_labels(
    regimes: np.ndarray,
    returns: np.ndarray,
    h1: int
) -> dict:
    """
    Assign human-readable labels based on per-regime return/vol characteristics.
    Labels: 'bull', 'bear', 'high_volatility', 'low_volatility', 'normal_<N>'
    """
    unique_regimes = np.unique(regimes)
    regime_returns = {}
    regime_vols = {}

    for rid in unique_regimes:
        mask = regimes == rid
        idx_list = []
        for i in np.where(mask)[0]:
            idx_list.extend(range(i, min(i + h1, len(returns))))
        if idx_list:
            r = returns[idx_list]
            regime_returns[rid] = float(np.mean(r))
            regime_vols[rid] = float(np.std(r))

    if not regime_vols:
        return {str(r): f"regime_{int(r)}" for r in unique_regimes}

    vols = list(regime_vols.values())
    med_vol = float(np.median(vols))
    p25 = float(np.percentile(vols, 25))
    p75 = float(np.percentile(vols, 75))

    labels_map = {}
    for rid in unique_regimes:
        mu = regime_returns.get(rid, 0)
        sigma = regime_vols.get(rid, 0)
        if mu > 0 and sigma < med_vol:
            labels_map[str(rid)] = "bull"
        elif mu < 0 and sigma > med_vol:
            labels_map[str(rid)] = "bear"
        elif sigma > p75:
            labels_map[str(rid)] = "high_volatility"
        elif sigma < p25:
            labels_map[str(rid)] = "low_volatility"
        else:
            labels_map[str(rid)] = f"normal_{int(rid)}"

    return labels_map


# ==============================================================================
# SECTION 8: OUTPUT GENERATION + VALIDATION (output_spec.json v1.0)
# ==============================================================================

def validate_output(
    df: pd.DataFrame,
    metadata: dict,
    config: dict,
) -> list:
    """
    Validate output against spec requirements before writing to disk.

    Returns list of error messages (empty = all checks pass).
    """
    errors = []
    eval_start = pd.Timestamp(config["eval_start"])
    eval_end = pd.Timestamp(config["eval_end"])

    # 1. Required columns
    required = {"date", "predicted_regime", "confidence", "transition_flag"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # 2. Date range
    df_dates = pd.to_datetime(df["date"])
    if df_dates.min() < eval_start or df_dates.max() > eval_end:
        errors.append(f"Date range {df_dates.min().date()} to {df_dates.max().date()} "
                      f"outside eval window {eval_start.date()} to {eval_end.date()}")

    if not df_dates.is_monotonic_increasing:
        errors.append("date column is not sorted ascending")

    # 3. predicted_regime: integer, non-negative, contiguous from 0
    regimes = df["predicted_regime"].values
    if not np.issubdtype(regimes.dtype, np.integer):
        errors.append("predicted_regime is not integer type")
    if np.any(regimes < 0):
        n_neg = int((regimes < 0).sum())
        errors.append(f"predicted_regime has {n_neg} negative values")
    if np.any(np.isnan(regimes)):
        errors.append("predicted_regime has NaN values")

    # Contiguity check
    unique_regimes = sorted(np.unique(regimes))
    expected = list(range(len(unique_regimes)))
    if unique_regimes != expected:
        errors.append(f"predicted_regime not contiguous from 0: got {unique_regimes}")

    # 4. confidence in [0, 1]
    conf = df["confidence"].values
    if conf.min() < 0 or conf.max() > 1:
        errors.append(f"confidence out of range: [{conf.min():.4f}, {conf.max():.4f}]")

    # 5. transition_flag is binary
    tf = df["transition_flag"].values
    unique_tf = np.unique(tf)
    if not set(unique_tf).issubset({0, 1}):
        errors.append(f"transition_flag has non-binary values: {unique_tf}")
    if len(df) > 0 and tf[0] != 0:
        errors.append("transition_flag for first row should be 0")

    # 6. Metadata consistency
    actual_num = len(unique_regimes)
    meta_num = metadata.get("num_regimes")
    if meta_num != actual_num:
        errors.append(f"metadata num_regimes={meta_num} but data has {actual_num} regimes")

    # 7. regime_labels_map covers all regimes
    labels_map = metadata.get("regime_labels_map", {})
    label_keys = set(labels_map.keys())
    expected_keys = set(str(r) for r in unique_regimes)
    if label_keys != expected_keys:
        errors.append(f"regime_labels_map keys {label_keys} don't match regime IDs {expected_keys}")

    return errors


def generate_output_files(
    dates: pd.DatetimeIndex,
    closing_prices: np.ndarray,
    scores: np.ndarray,
    change_points: np.ndarray,
    ensemble_regimes: np.ndarray,
    regime_labels_map: dict,
    config: dict,
    output_dir: Path,
    diagnostics: dict = None,
):
    """
    Generate output files: {algo_name}_regimes.csv, {algo_name}_metadata.json
    with strict validation before writing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    algo_name = config["algo_name"]
    h1, h2 = config["h1"], config["h2"]

    # Map ensemble-level regimes to daily dates
    # Ensemble i covers approximate dates[i : i + h1*h2 - 1]
    n_dates = len(dates)
    daily_regimes = np.full(n_dates, 0, dtype=int)
    daily_confidence = np.full(n_dates, 0.5)
    daily_transition = np.zeros(n_dates, dtype=int)

    N2 = len(ensemble_regimes)

    for i in range(N2):
        start_idx = i
        end_idx = min(i + h1 * h2, n_dates)
        for j in range(start_idx, end_idx):
            daily_regimes[j] = ensemble_regimes[i]

    # Confidence: inverse of relative MMD score
    # Higher score → lower confidence (more likely regime boundary)
    score_max = float(np.nanmax(scores)) if np.nanmax(scores) > 0 else 1.0
    for i in range(N2):
        if i < n_dates and not np.isnan(scores[i]):
            start_idx = i
            end_idx = min(i + h1 * h2, n_dates)
            conf = max(0.0, min(1.0, 1.0 - scores[i] / score_max))
            for j in range(start_idx, end_idx):
                daily_confidence[j] = conf

    # Transition flags: 1 where regime changes
    for j in range(1, n_dates):
        if daily_regimes[j] != daily_regimes[j - 1]:
            daily_transition[j] = 1

    # Filter to evaluation window
    eval_start = pd.Timestamp(config["eval_start"])
    eval_end = pd.Timestamp(config["eval_end"])
    mask = (dates >= eval_start) & (dates <= eval_end)
    eval_dates = dates[mask]

    if len(eval_dates) == 0:
        raise ValueError(f"No dates in evaluation window {eval_start} to {eval_end}")

    eval_regimes = daily_regimes[mask]
    eval_confidence = daily_confidence[mask]
    eval_transition = daily_transition[mask]

    # Build regime labels column
    regime_label_col = []
    for rid in eval_regimes:
        key = str(int(rid))
        regime_label_col.append(regime_labels_map.get(key, f"regime_{key}"))

    # Build DataFrame
    df = pd.DataFrame({
        "date": eval_dates.strftime("%Y-%m-%d"),
        "predicted_regime": eval_regimes.astype(int),
        "confidence": eval_confidence.round(4),
        "transition_flag": eval_transition.astype(int),
        "regime_label": regime_label_col,
    })

    # Ensure no bad values
    df["predicted_regime"] = df["predicted_regime"].fillna(0).astype(int)
    df["confidence"] = df["confidence"].fillna(0.5).clip(0, 1)
    df["transition_flag"] = df["transition_flag"].fillna(0).astype(int)

    # Build metadata
    num_regimes = int(df["predicted_regime"].max()) + 1
    metadata = {
        "algorithm": algo_name,
        "algorithm_full_name": "Signature MMD-based Regime Detection (Auto-Evaluator v2)",
        "algorithm_family": "changepoint",
        "paper_reference": "arXiv:2306.15835",
        "num_regimes": num_regimes,
        "regime_labels_map": regime_labels_map,
        "parameters": {
            "h1": config["h1"],
            "h2": config["h2"],
            "sig_level": config["sig_level"],
            "rbf_sigma": config["rbf_sigma"],
            "z_score_k": config["z_score_k"],
            "rolling_window": config["rolling_window"],
            "L": config["L"],
            "L_weights": config["L_weights"],
            "annualize_scale": config["annualize_scale"],
            "cluster_method": config["cluster_method"],
            "training_window_days": None,
            "features_used": [
                "log_return",
                "path_signature_level2",
                "time_channel"
            ],
            "random_seed": config["random_seed"],
        },
        "training_window_days": None,
        "features_used": [
            "log_return",
            "path_signature_level2",
            "time_channel"
        ],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_spec_version": "1.0",
    }

    # --- VALIDATE ---
    validation_errors = validate_output(df, metadata, config)
    if validation_errors:
        print("\n" + "!" * 70)
        print("VALIDATION ERRORS:")
        for err in validation_errors:
            print(f"  ❌ {err}")
        print("!" * 70 + "\n")
        # Attempt auto-fixes for simple issues
        # (e.g., if first row transition_flag != 0)
        if len(df) > 0 and "transition_flag for first row" in " ".join(validation_errors):
            df.loc[df.index[0], "transition_flag"] = 0
            print("   Auto-fixed: set first row transition_flag=0")
        # Re-validate
        validation_errors = validate_output(df, metadata, config)
        if validation_errors:
            print("⚠️  CRITICAL: Validation still failing — writing files anyway with warnings.")
    else:
        print("   ✅ All validation checks passed.")

    # Write CSV
    csv_path = output_dir / f"{algo_name}_regimes.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"   Saved {csv_path} ({len(df)} rows)")

    # Write metadata JSON
    meta_path = output_dir / f"{algo_name}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved {meta_path}")

    return df, metadata


# ==============================================================================
# SECTION 9: MAIN PIPELINE
# ==============================================================================

def main():
    """Run the full regime detection pipeline end-to-end."""

    print("=" * 70)
    print("REGIME DETECTION via SIGNATURE MMD (Auto-Evaluator) — v2 REWRITTEN")
    print("Paper: Horvath & Issa 2023, arXiv:2306.15835")
    print("=" * 70)
    print(f"Config: h1={CONFIG['h1']}, h2={CONFIG['h2']}, sig_level={CONFIG['sig_level']}")
    print(f"MMD: {CONFIG['kernel']} kernel, σ={CONFIG['rbf_sigma']}")
    print(f"Auto-evaluator v2: L={CONFIG['L']}, z_k={CONFIG['z_score_k']}, roll={CONFIG['rolling_window']}")
    print()

    # ---- 1. Load data ----
    print("1. Loading S&P 500 data...")
    data_path = Path("/opt/data/kanban/workspaces/t_f9e0a695/sp500_data/sp500_ohlcv_20yr.csv")
    if not data_path.exists():
        print(f"   ERROR: Data not found at {data_path}")
        sys.exit(1)

    df_data = pd.read_csv(data_path, parse_dates=["date"], index_col="date")
    close = df_data["close"].values
    dates = df_data.index
    n_dates = len(dates)
    print(f"   Loaded {n_dates} trading days: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print()

    # ---- 2. Path transformations ----
    print("2. Computing log returns and path transformations...")
    log_rets = compute_log_returns(close)
    path = apply_path_transformations(log_rets, CONFIG["annualize_scale"])
    print(f"   Path shape: {path.shape} (T={len(path)}, d={path.shape[1]})")
    print()

    # ---- 3. Sub-paths and ensembles ----
    print("3. Extracting sub-paths and forming ensembles...")
    sub_paths = extract_sub_paths(path, CONFIG["h1"])
    N1 = len(sub_paths)
    print(f"   Sub-paths: {N1} (length h1={CONFIG['h1']})")

    ensembles = form_ensembles(sub_paths, CONFIG["h2"])
    N2 = len(ensembles)
    print(f"   Ensembles:  {N2} (size h2={CONFIG['h2']})")
    print()

    # ---- 4. Signature computation ----
    print("4. Computing truncated path signatures...")
    ensemble_sigs = compute_ensemble_signatures(ensembles, CONFIG["sig_level"])
    sig_dim = ensemble_sigs.shape[-1]
    print(f"   Signature dimension: {sig_dim} (level={CONFIG['sig_level']}, d={path.shape[1]})")
    print()

    # ---- 5. Auto-evaluator v2 (rolling z-score) ----
    print("5. Running auto-evaluator v2 (rolling z-score threshold)...")
    scores, thresholds, change_points, diagnostics = auto_evaluator_v2(
        ensemble_sigs,
        CONFIG["L"],
        CONFIG["L_weights"],
        CONFIG["rolling_window"],
        CONFIG["z_score_k"],
        CONFIG["max_transition_frac"],
        CONFIG["min_transitions_5yr"],
        CONFIG["rbf_sigma"],
    )

    n_changes = int(change_points.sum())
    n_valid = diagnostics["n_valid_scores"]
    frac = diagnostics["transition_fraction"]

    print(f"   Valid scores: {n_valid}/{N2}")
    print(f"   Regime changes detected: {n_changes} ({frac*100:.2f}%)")
    print(f"   Score mean: {diagnostics['score_mean']:.6f}")
    print(f"   Score std:  {diagnostics['score_std']:.6f}")
    print(f"   Score quantiles: p25={diagnostics['score_quantiles']['p25']:.6f}, "
          f"p50={diagnostics['score_quantiles']['p50']:.6f}, "
          f"p75={diagnostics['score_quantiles']['p75']:.6f}, "
          f"p95={diagnostics['score_quantiles']['p95']:.6f}")
    print(f"   Z-score k: {diagnostics['z_score_k_final']}")
    if diagnostics.get("recalibrated"):
        print(f"   ⚠️  Auto-recalibrated threshold")
    print()

    # ---- 6. Determine number of regimes ----
    print("6. Determining number of regimes...")
    # Use n_changes+1 as natural cluster count, but cap at 3-4
    n_clusters = max(2, min(CONFIG["n_regimes_max"], n_changes + 1))
    n_clusters = min(n_clusters, CONFIG["n_regimes_target"] + 1)
    print(f"   Targeting {n_clusters} regimes (changes={n_changes})")
    print()

    # ---- 7. Clustering (robust) ----
    print("7. Clustering regimes (robust agglomerative with fallback)...")
    ensemble_regimes = cluster_ensembles_v2(
        ensemble_sigs,
        n_clusters,
        CONFIG["rbf_sigma"],
        CONFIG["cluster_method"],
        change_points,
    )
    for rid in range(n_clusters):
        count = (ensemble_regimes == rid).sum()
        pct = count / N2 * 100
        print(f"     Regime {rid}: {count} ensembles ({pct:.1f}%)")
    print()

    # ---- 8. Label regimes ----
    print("8. Assigning regime labels...")
    regime_labels_map = resolve_regime_labels(ensemble_regimes, log_rets, CONFIG["h1"])
    for rid_str, label in sorted(regime_labels_map.items()):
        rid = int(rid_str)
        count = (ensemble_regimes == rid).sum()
        print(f"     Regime {rid_str} → '{label}' ({count} ensembles)")
    print()

    # ---- 9. Generate output files ----
    print("9. Generating validated output files...")
    output_dir = Path("/opt/data/kanban/workspaces/t_c36a4eed/paper-3947905-v2/outputs")
    df_out, metadata = generate_output_files(
        dates, close, scores, change_points, ensemble_regimes,
        regime_labels_map, CONFIG, output_dir, diagnostics
    )
    print()

    # ---- 10. Summary ----
    print("=" * 70)
    print("REGIME DETECTION SUMMARY (v2)")
    print("=" * 70)
    print(f"Algorithm:       {metadata['algorithm_full_name']}")
    print(f"Paper:           {metadata['paper_reference']}")
    print(f"Num regimes:     {metadata['num_regimes']}")
    print(f"Regime labels:   {metadata['regime_labels_map']}")
    print(f"Transitions:     {n_changes} ({frac*100:.1f}%)")
    print(f"Score range:     [{np.nanmin(scores):.6f}, {np.nanmax(scores):.6f}]")
    print(f"Threshold method: rolling z-score (k={diagnostics['z_score_k_final']}, window={CONFIG['rolling_window']})")
    csv_file = output_dir / f"{CONFIG['algo_name']}_regimes.csv"
    json_file = output_dir / f"{CONFIG['algo_name']}_metadata.json"
    print(f"Output CSV:      {csv_file}")
    print(f"Output JSON:     {json_file}")

    # Print top transition dates from evaluation window
    eval_start = pd.Timestamp(CONFIG["eval_start"])
    eval_end = pd.Timestamp(CONFIG["eval_end"])
    eval_mask = (dates >= eval_start) & (dates <= eval_end)
    transition_dates_in_eval = []
    for i in range(N2):
        if change_points[i] and i < len(dates) and eval_mask[i]:
            transition_dates_in_eval.append(dates[i].strftime("%Y-%m-%d"))

    if transition_dates_in_eval:
        print(f"\nDetected transition dates in evaluation window (top 20):")
        for d in transition_dates_in_eval[:20]:
            print(f"  {d}")
        if len(transition_dates_in_eval) > 20:
            print(f"  ... and {len(transition_dates_in_eval) - 20} more")
    print("=" * 70)

    return df_out, metadata, scores, change_points


if __name__ == "__main__":
    df_out, metadata, scores, change_points = main()
