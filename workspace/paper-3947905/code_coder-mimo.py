"""
Regime Detection Implementation based on:
"Non-parametric online market regime detection and regime clustering
for multidimensional and path-dependent data structures"
(Horvath & Issa, 2023) - arXiv:2306.15835

This implementation uses:
1. Path signature features on rolling windows
2. MMD-based two-sample test with RBF kernel
3. Online change-point detection via L-lag auto evaluation score
4. Post-hoc regime clustering via k-means

Author: coder-mimo (kanban task t_19cc81ac)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Tuple, List, Dict, Optional

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# Hyperparameters (from paper Section 6.1, adapted for daily data)
H1 = 8          # Sub-path length (trading days per window)
H2 = 8          # Ensemble size (number of windows in ensemble)
LAGS = [1, 2, 4, 8]  # L = {1, 2, 4, 8} for auto evaluation
LAG_WEIGHTS = {1: 0.4, 2: 0.3, 4: 0.2, 8: 0.1}  # Exponential decay weights
ALPHA = 0.95    # Detection threshold quantile
PRIOR_WINDOW = 200  # Number of historical MMD scores for empirical prior
SIGNATURE_ORDER = 3  # Truncated signature order N
RBF_SIGMA = 1.0  # RBF kernel bandwidth
N_REGIMES = 3   # Number of regimes to detect (bull/bear/sideways)
MIN_SEGMENT_LEN = 10  # Minimum days for a regime segment

# Data paths
SP500_DATA_PATH = "/opt/data/kanban/workspaces/t_f9e0a695/sp500_data/sp500_ohlcv_20yr.csv"
OUTPUT_DIR = "/opt/data/kanban/workspaces/t_19cc81ac/paper-3947905"


# ─────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────
def compute_features(prices: pd.Series) -> pd.DataFrame:
    """
    Compute financial features from price series.
    
    Features used (following output_spec.json naming):
    - daily_return: log returns
    - daily_vol_20d: 20-day rolling volatility
    - cumulative_return_20d: 20-day cumulative return
    - rsi_14d: 14-day RSI
    """
    df = pd.DataFrame(index=prices.index)
    
    # Log returns (Eq. for path increments)
    df["daily_return"] = np.log(prices / prices.shift(1))
    
    # Rolling volatility (20-day)
    df["daily_vol_20d"] = df["daily_return"].rolling(window=20, min_periods=10).std()
    
    # Cumulative return (20-day)
    df["cumulative_return_20d"] = np.log(prices / prices.shift(20))
    
    # RSI (14-day)
    delta = df["daily_return"]
    gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=7).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))
    
    return df


# ─────────────────────────────────────────────────────────────────────
# Truncated Path Signature (Def 2.1, Eq 5-6)
# ─────────────────────────────────────────────────────────────────────
def truncated_signature(path: np.ndarray, order: int = SIGNATURE_ORDER) -> np.ndarray:
    """
    Compute truncated path signature up to given order.
    
    For a 1D path, the signature terms are:
    Level 0: 1
    Level 1: integral dX = X[b] - X[a]  (increment)
    Level 2: integral X dX  (area under path)
    Level 3: integral X^2 dX  (cubic term)
    
    For multi-dim paths, we get tensor products.
    
    Args:
        path: 1D array of path values (e.g., cumulative log returns)
        order: truncation level N
    
    Returns:
        Signature vector of dimension (1 + d + d^2 + ... + d^N) where d = path dim
    """
    d = 1  # Univariate (we augment with time below)
    n = len(path)
    
    # Augment path with normalized time (paper uses time-augmented paths)
    t = np.linspace(0, 1, n)
    # Create 2D path: [time, value]
    path_2d = np.column_stack([t, path])
    d_aug = 2  # augmented dimension
    
    sig_components = [1.0]  # Level 0: always 1
    
    # Level 1: increments for each dimension
    increments = path_2d[-1] - path_2d[0]
    sig_components.extend(increments.tolist())
    
    if order >= 2:
        # Level 2: iterated integrals (area terms)
        # For 2D augmented path, we compute:
        # integral_0^1 X^i_s dX^j_s for all (i,j) pairs
        dt = np.diff(t)
        for i in range(d_aug):
            for j in range(d_aug):
                # Left-point Riemann sum approximation
                integral = np.sum(path_2d[:-1, i] * np.diff(path_2d[:, j]))
                sig_components.append(integral)
    
    if order >= 3:
        # Level 3: third-order iterated integrals
        for i in range(d_aug):
            for j in range(d_aug):
                for k in range(d_aug):
                    integral = 0.0
                    for t_idx in range(n - 2):
                        dt1 = t[t_idx + 1] - t[t_idx]
                        dt2 = t[t_idx + 2] - t[t_idx + 1]
                        integral += (path_2d[t_idx, i] * path_2d[t_idx, j] *
                                   (path_2d[t_idx + 2, k] - path_2d[t_idx + 1, k]))
                    sig_components.append(integral)
    
    return np.array(sig_components)


# ─────────────────────────────────────────────────────────────────────
# RBF Kernel and MMD (Section 2.3, Eq 16)
# ─────────────────────────────────────────────────────────────────────
def rbf_kernel(X: np.ndarray, Y: np.ndarray, sigma: float = RBF_SIGMA) -> np.ndarray:
    """
    Compute RBF (Gaussian) kernel matrix between X and Y.
    
    k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    """
    sq_dists = cdist(X, Y, "sqeuclidean")
    return np.exp(-sq_dists / (2 * sigma ** 2))


def mmd_unbiased(X: np.ndarray, Y: np.ndarray, sigma: float = RBF_SIGMA) -> float:
    """
    Compute unbiased MMD^2 estimator (Eq 16).
    
    D_u^k(P,Q)^2 = 1/(n(n-1)) sum_{i!=j} k(xi,xj)
                   - 2/(mn) sum_i sum_j k(xi,yj)
                   + 1/(m(m-1)) sum_{i!=j} k(yi,yj)
    
    Args:
        X: samples from distribution P, shape (n, d)
        Y: samples from distribution Q, shape (m, d)
        sigma: RBF kernel bandwidth
    
    Returns:
        MMD^2 estimate (can be negative due to unbiasedness; clipped to 0)
    """
    n = X.shape[0]
    m = Y.shape[0]
    
    if n < 2 or m < 2:
        return 0.0
    
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    
    # Unbiased: exclude diagonal
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)
    
    term1 = Kxx.sum() / (n * (n - 1))
    term2 = -2 * Kxy.sum() / (m * n)
    term3 = Kyy.sum() / (m * (m - 1))
    
    mmd2 = term1 + term2 + term3
    return max(0.0, mmd2)  # Clip negative values


def mmd_signature_ensemble(
    ensemble1: List[np.ndarray],
    ensemble2: List[np.ndarray],
    sigma: float = RBF_SIGMA
) -> float:
    """
    Compute MMD between two ensembles of sub-paths using signature features.
    
    Paper approach: compute truncated signatures of each sub-path,
    then compute RBF MMD between signature vectors.
    
    Args:
        ensemble1: list of 1D arrays (sub-paths)
        ensemble2: list of 1D arrays (sub-paths)
        sigma: RBF kernel bandwidth
    
    Returns:
        MMD^2 between the two ensembles
    """
    sigs1 = np.array([truncated_signature(sp) for sp in ensemble1])
    sigs2 = np.array([truncated_signature(sp) for sp in ensemble2])
    
    return mmd_unbiased(sigs1, sigs2, sigma)


# ─────────────────────────────────────────────────────────────────────
# Online Regime Detection (Section 4, Eq 29)
# ─────────────────────────────────────────────────────────────────────
def compute_subpaths(returns: np.ndarray, h1: int) -> List[np.ndarray]:
    """
    Decompose returns into overlapping sub-paths of length h1 (Def 3.1).
    
    Each sub-path is a segment of cumulative log returns (path-like).
    """
    n = len(returns)
    subpaths = []
    for i in range(n - h1 + 1):
        # Sub-path is the cumulative returns in the window
        segment = returns[i:i + h1]
        cumret = np.cumsum(segment)  # Convert to path (cumulative)
        subpaths.append(cumret)
    return subpaths


def auto_evaluation_score(
    subpaths: List[np.ndarray],
    lags: List[int],
    weights: Dict[int, float],
    h2: int,
    sigma: float = RBF_SIGMA
) -> np.ndarray:
    """
    Compute L-lag auto evaluation score vector (Eq 29).
    
    A_L(s_hat)_i = sum_{l in L} w_l * D_sig^r(s_{i-l}, s_i)
    
    For each ensemble index i, compare the current ensemble (h2 consecutive
    sub-paths ending at i) against lagged ensembles.
    
    Args:
        subpaths: list of sub-path arrays
        lags: list of lag values L
        weights: dict mapping lag to weight w_l
        h2: ensemble size
        sigma: RBF kernel bandwidth
    
    Returns:
        Score vector A_L
    """
    N = len(subpaths)
    max_lag = max(lags)
    
    # We need at least h2 + max_lag sub-paths
    start_idx = h2 + max_lag - 1
    if start_idx >= N:
        return np.array([])
    
    scores = np.zeros(N)
    
    for i in range(start_idx, N):
        # Current ensemble: h2 sub-paths ending at i
        ensemble_i = subpaths[max(0, i - h2 + 1):i + 1]
        
        weighted_score = 0.0
        for lag in lags:
            j = i - lag
            if j - h2 + 1 < 0:
                continue
            
            # Lagged ensemble
            ensemble_j = subpaths[max(0, j - h2 + 1):j + 1]
            
            # Compute MMD between ensembles
            mmd = mmd_signature_ensemble(ensemble_i, ensemble_j, sigma)
            weighted_score += weights.get(lag, 0.0) * np.sqrt(mmd)
        
        scores[i] = weighted_score
    
    return scores


def detect_changepoints(
    scores: np.ndarray,
    alpha: float = ALPHA,
    prior_window: int = PRIOR_WINDOW,
    min_segment_len: int = MIN_SEGMENT_LEN
) -> np.ndarray:
    """
    Detect change points from auto evaluation scores.
    
    Rule: flag time i as regime change if A_L[i] > quantile(prior, alpha)
    where prior = scores[max(0, i-prior_window):i]
    
    Args:
        scores: auto evaluation score vector
        alpha: quantile threshold
        prior_window: size of empirical prior window
        min_segment_len: minimum days between change points
    
    Returns:
        Boolean array indicating detected change points
    """
    n = len(scores)
    flags = np.zeros(n, dtype=bool)
    
    for i in range(prior_window, n):
        # Empirical prior from recent history
        prior_scores = scores[max(0, i - prior_window):i]
        prior_scores = prior_scores[prior_scores > 0]  # Exclude zeros
        
        if len(prior_scores) < 10:
            continue
        
        threshold = np.quantile(prior_scores, alpha)
        
        if scores[i] > threshold:
            # Check minimum segment length from last change point
            last_change = np.where(flags[:i])[0]
            if len(last_change) == 0 or (i - last_change[-1]) >= min_segment_len:
                flags[i] = True
    
    return flags


# ─────────────────────────────────────────────────────────────────────
# Regime Clustering (Section 5)
# ─────────────────────────────────────────────────────────────────────
def assign_regime_labels(
    returns: pd.Series,
    change_points: np.ndarray,
    dates: pd.DatetimeIndex,
    n_regimes: int = N_REGIMES
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Assign regime labels to each day based on change points.
    
    1. Segment the time series at change points
    2. Compute features for each segment
    3. Cluster segments into n_regimes using k-means
    4. Assign labels
    
    Args:
        returns: daily log returns
        change_points: boolean array of change point flags
        dates: trading dates
        n_regimes: number of regimes to detect
    
    Returns:
        (regime_labels, regime_label_map)
    """
    n = len(returns)
    
    # Create segments from change points
    cp_indices = np.where(change_points)[0]
    
    if len(cp_indices) == 0:
        # No change points detected: single regime
        return np.zeros(n, dtype=int), {0: "single_regime"}
    
    # Build segments: [start, end) for each regime period
    segment_starts = [0] + cp_indices.tolist()
    segment_ends = cp_indices.tolist() + [n]
    
    # Compute segment features for clustering
    segment_features = []
    for start, end in zip(segment_starts, segment_ends):
        seg_returns = returns.iloc[start:end]
        if len(seg_returns) < 2:
            features = [0, 0, 0, 0]
        else:
            features = [
                seg_returns.mean(),           # Mean return
                seg_returns.std(),            # Volatility
                seg_returns.skew() if len(seg_returns) > 2 else 0,  # Skewness
                seg_returns.kurtosis() if len(seg_returns) > 3 else 0  # Kurtosis
            ]
        segment_features.append(features)
    
    segment_features = np.array(segment_features)
    
    # Handle NaN/Inf
    segment_features = np.nan_to_num(segment_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Cluster segments
    actual_regimes = min(n_regimes, len(segment_starts))
    if actual_regimes < 2:
        labels_per_segment = np.zeros(len(segment_starts), dtype=int)
    else:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(segment_features)
        
        kmeans = KMeans(n_clusters=actual_regimes, random_state=SEED, n_init=10)
        labels_per_segment = kmeans.fit_predict(features_scaled)
    
    # Map segments to days
    regime_labels = np.zeros(n, dtype=int)
    for idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        regime_labels[start:end] = labels_per_segment[idx]
    
    # Create label map based on mean return of each regime
    regime_returns = {}
    for r in range(actual_regimes):
        mask = regime_labels == r
        if mask.sum() > 0:
            regime_returns[r] = returns.values[mask].mean()
    
    # Sort regimes by return: lowest = bear, highest = bull
    sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])
    
    label_names = ["bear", "sideways", "bull", "strong_bull", "strong_bear"]
    # Create remapping: old_label -> new_label (contiguous 0,1,2... sorted by return)
    remap = {}
    regime_label_map = {}
    for rank, (orig_label, _) in enumerate(sorted_regimes):
        remap[orig_label] = rank
        name_idx = min(rank, len(label_names) - 1) if len(sorted_regimes) <= len(label_names) else rank
        regime_label_map[rank] = label_names[name_idx] if name_idx < len(label_names) else f"regime_{rank}"
    
    # Apply remapping to regime_labels
    new_labels = np.zeros(n, dtype=int)
    for old_label, new_label in remap.items():
        new_labels[regime_labels == old_label] = new_label
    regime_labels = new_labels
    
    return regime_labels, regime_label_map


# ─────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────
def run_regime_detection():
    """
    Full pipeline:
    1. Load S&P 500 data
    2. Compute features
    3. Run online regime detection via signature MMD
    4. Assign regime labels
    5. Output in required format
    """
    print("=" * 70)
    print("REGIME DETECTION: arXiv 2306.15835 Implementation")
    print("Non-parametric online market regime detection")
    print("using path signatures and MMD")
    print("=" * 70)
    print()
    
    # ── 1. Load data ──
    print("[1/6] Loading S&P 500 data...")
    df = pd.read_csv(SP500_DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    
    print(f"  Loaded {len(df)} trading days")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print()
    
    # ── 2. Compute features ──
    print("[2/6] Computing features...")
    features = compute_features(df["close"])
    features = features.dropna()
    
    print(f"  Features computed: {list(features.columns)}")
    print(f"  Valid samples: {len(features)}")
    print()
    
    # ── 3. Build sub-paths from log returns ──
    print("[3/6] Building sub-paths and computing signatures...")
    returns = features["daily_return"]
    
    subpaths = compute_subpaths(returns.values, H1)
    print(f"  Sub-paths created: {len(subpaths)} (window size = {H1})")
    print(f"  Ensemble size: {H2}")
    print(f"  Signature truncation order: {SIGNATURE_ORDER}")
    print()
    
    # ── 4. Run auto evaluation (online detection) ──
    print("[4/6] Running L-lag auto evaluation (online regime detection)...")
    print(f"  Lags: {LAGS}")
    print(f"  Weights: {LAG_WEIGHTS}")
    print(f"  Alpha (threshold quantile): {ALPHA}")
    print(f"  Prior window: {PRIOR_WINDOW}")
    
    # Run on the evaluation window (2021-05-10 to 2026-05-06 per output spec)
    # But first detect on full history for training, then focus on eval window
    scores = auto_evaluation_score(subpaths, LAGS, LAG_WEIGHTS, H2, RBF_SIGMA)
    print(f"  Score vector length: {len(scores)}")
    print(f"  Non-zero scores: {(scores > 0).sum()}")
    print()
    
    # ── 5. Detect change points ──
    print("[5/6] Detecting change points...")
    change_points = detect_changepoints(scores, ALPHA, PRIOR_WINDOW, MIN_SEGMENT_LEN)
    
    # Map change_points from subpath space to returns space
    # Subpaths are created from returns[H1-1:] onwards, and scores start at offset
    score_offset = H2 + max(LAGS) - 1  # offset in subpath space
    # Subpath i corresponds to returns index (H1 - 1 + i)
    # Score i corresponds to subpath (score_offset + i)
    # So score i corresponds to returns index (H1 - 1 + score_offset + i)
    
    # Build change_points aligned to returns index
    full_cp = np.zeros(len(returns), dtype=bool)
    for i in range(len(change_points)):
        if change_points[i]:
            ret_idx = (H1 - 1) + score_offset + i
            if ret_idx < len(returns):
                full_cp[ret_idx] = True
    
    n_changes = full_cp.sum()
    print(f"  Detected {n_changes} regime transitions")
    if n_changes > 0:
        cp_dates = returns.index[full_cp]
        print(f"  First transition: {cp_dates[0].date()}")
        print(f"  Last transition: {cp_dates[-1].date()}")
    print()
    
    # ── 6. Assign regime labels ──
    print("[6/6] Assigning regime labels...")
    regime_labels, regime_map = assign_regime_labels(
        returns, full_cp, returns.index, N_REGIMES
    )
    
    print(f"  Regimes detected: {len(regime_map)}")
    for k, v in regime_map.items():
        mask = regime_labels == k
        pct = mask.sum() / len(regime_labels) * 100
        print(f"    Regime {k} ({v}): {mask.sum()} days ({pct:.1f}%)")
    print()
    
    # ── Build output dataframe ──
    # Align to full date range (some features may have NaN at start)
    full_dates = df.index
    full_regime = np.full(len(full_dates), -1, dtype=int)
    full_confidence = np.full(len(full_dates), 0.0)
    full_transition = np.zeros(len(full_dates), dtype=int)
    
    # Align scores to returns space for confidence computation
    # regime_labels is aligned to returns.index, full_dates is the full price index
    # Map regime_labels to full_regime using date alignment
    returns_start_loc = full_dates.get_loc(returns.index[0])
    for i in range(len(regime_labels)):
        full_idx = returns_start_loc + i
        if full_idx < len(full_regime):
            full_regime[full_idx] = regime_labels[i]
    
    # Confidence from scores
    valid_scores = scores[scores > 0]
    if len(valid_scores) > 0:
        score_max = np.percentile(valid_scores, 99)
        for i in range(len(scores)):
            if scores[i] > 0:
                ret_idx = (H1 - 1) + score_offset + i
                if ret_idx < len(returns):
                    full_idx = returns_start_loc + ret_idx
                    if full_idx < len(full_dates):
                        full_confidence[full_idx] = min(1.0, scores[i] / score_max) if score_max > 0 else 0.5
    
    # Set default confidence for labeled days
    mask_labeled = full_regime >= 0
    mask_zero_conf = (full_confidence == 0) & mask_labeled
    full_confidence[mask_zero_conf] = 0.5
    
    # Transition flags
    for i in range(1, len(full_regime)):
        if full_regime[i] >= 0 and full_regime[i - 1] >= 0:
            if full_regime[i] != full_regime[i - 1]:
                full_transition[i] = 1
    
    # First row: transition_flag = 0 per spec
    full_transition[0] = 0
    
    # Create labels
    full_labels = []
    for r in full_regime:
        if r in regime_map:
            full_labels.append(regime_map[r])
        else:
            full_labels.append("")
    
    # Build output dataframe
    out_df = pd.DataFrame({
        "date": full_dates.strftime("%Y-%m-%d"),
        "predicted_regime": full_regime,
        "confidence": np.round(full_confidence, 4),
        "transition_flag": full_transition,
        "regime_label": full_labels
    })
    
    # ── Print summary ──
    print("=" * 70)
    print("REGIME DETECTION RESULTS SUMMARY")
    print("=" * 70)
    
    # Show transitions
    transitions = out_df[out_df["transition_flag"] == 1].copy()
    print(f"\nTotal regime transitions: {len(transitions)}")
    if len(transitions) > 0:
        print("\nTransition details:")
        for _, row in transitions.head(20).iterrows():
            prev_regime = out_df.loc[out_df.index[out_df["date"] == row["date"]][0] - 1, "predicted_regime"] if out_df.index[out_df["date"] == row["date"]][0] > 0 else "?"
            print(f"  {row['date']}: regime {prev_regime} -> {row['predicted_regime']} ({row['regime_label']})")
    
    # Regime distribution in eval window
    eval_mask = (out_df["date"] >= "2021-05-10") & (out_df["date"] <= "2026-05-06")
    eval_df = out_df[eval_mask]
    print(f"\nEvaluation window regime distribution:")
    for regime_id in sorted(eval_df["predicted_regime"].unique()):
        count = (eval_df["predicted_regime"] == regime_id).sum()
        label = regime_map.get(regime_id, "unknown")
        print(f"  Regime {regime_id} ({label}): {count} days")
    
    # ── Save output ──
    csv_path = os.path.join(OUTPUT_DIR, "sig_mmd_regimes.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\nCSV output saved to: {csv_path}")
    
    # Metadata JSON
    metadata = {
        "algorithm": "sig_mmd_online",
        "algorithm_full_name": "Signature MMD Online Regime Detection (Horvath & Issa 2023)",
        "algorithm_family": "changepoint",
        "paper_reference": "arXiv:2306.15835",
        "num_regimes": int(len(regime_map)),
        "regime_labels_map": {str(k): v for k, v in regime_map.items()},
        "parameters": {
            "h1": H1,
            "h2": H2,
            "lags": LAGS,
            "lag_weights": LAG_WEIGHTS,
            "alpha": ALPHA,
            "prior_window": PRIOR_WINDOW,
            "signature_order": SIGNATURE_ORDER,
            "rbf_sigma": RBF_SIGMA,
            "min_segment_len": MIN_SEGMENT_LEN,
            "training_window_days": None,
            "features_used": ["daily_return", "daily_vol_20d", "cumulative_return_20d", "rsi_14d"],
            "random_seed": SEED
        },
        "training_window_days": None,
        "features_used": ["daily_return", "daily_vol_20d", "cumulative_return_20d", "rsi_14d"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_spec_version": "1.0"
    }
    
    meta_path = os.path.join(OUTPUT_DIR, "sig_mmd_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {meta_path}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    
    return out_df, metadata


if __name__ == "__main__":
    run_regime_detection()
