#!/usr/bin/env python3
"""
Non-parametric Online Market Regime Detection (v2 REWRITE)
Implementation of arXiv:2306.15835 (Horvath & Issa, 2023)

Key changes from v1:
  - Correct evaluation window: 2021-05-10 to end of available data
  - Produces spec-compliant CSV + JSON output
  - Uses S&P 500 OHLCV parquet data (not yfinance)
  - Signature-kernel MMD regime detection → contiguous 0-indexed regime IDs
  - Post-processing: merge short segments, cluster into 3-5 meaningful regimes
  - Fallback to volatility-based detection if MMD fails
  - Full validation pipeline before final write
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from datetime import datetime, timezone
from scipy import stats
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "/opt/data/kanban/workspaces/t_f9e0a695/sp500_data/sp500_ohlcv_20yr.parquet"
OUTPUT_DIR = "/opt/data/kanban/workspaces/t_1b79a957/paper-3947905-v2"
EVAL_START = "2021-05-10"
EVAL_END = "2025-12-31"  # data ends here; use all available

# Algorithm hyperparameters
H1 = 10          # sub-path length (days)
H2 = 8           # ensemble window (# sub-paths)
TRUNCATION = 2   # signature truncation level
RBF_SIGMA = 1.0  # RBF kernel bandwidth
LAGS = [1, 2]    # lag orders for auto-evaluator
MEMORY_WINDOW = 100
ALPHA = 0.05     # significance level
SEED = 42
ALGO_NAME = "sig_mmd_horvath_issa"

# Post-processing
MIN_SEGMENT_DAYS = 15     # merge segments shorter than this (3 trading weeks)
TARGET_N_REGIMES = 4      # desired number of regimes after clustering
REGIME_CLUSTER_FEATURES = ['daily_return', 'daily_vol_20d']  # features for clustering


# ============================================================================
# PATH TRANSFORMATIONS (Sec 3.1)
# ============================================================================

def apply_transforms(path: np.ndarray, transforms: List[str] = None) -> np.ndarray:
    if transforms is None:
        transforms = ['time', 'state']
    result = path.copy()
    for t in transforms:
        if t == 'time':
            t_vals = result[:, 0]
            denom = t_vals[-1] - t_vals[0]
            if abs(denom) < 1e-10:
                result[:, 0] = 0.0
            else:
                result[:, 0] = (t_vals - t_vals[0]) / denom
        elif t == 'state':
            for i in range(1, result.shape[1]):
                if abs(result[0, i]) > 1e-10:
                    result[:, i] /= result[0, i]
        elif t == 'incr':
            increments = np.abs(np.diff(result[:, 1:], axis=0))
            cum_increments = np.concatenate([[0], np.cumsum(np.sum(increments, axis=1))])
            result = np.column_stack([result, cum_increments + result[0, 1]])
    return result


# ============================================================================
# TRUNCATED SIGNATURE FEATURES
# ============================================================================

def compute_signature_features(path: np.ndarray, truncation: int = 2) -> np.ndarray:
    n, d = path.shape
    increments = np.diff(path, axis=0)
    features = [np.array([1.0])]
    features.append(path[-1] - path[0])  # Level 1
    if truncation >= 2:
        area = np.zeros(d)
        for dim in range(d):
            area[dim] = 0.5 * np.sum(
                (path[1:, dim] + path[:-1, dim]) * np.diff(path[:, 0])
            )
        features.append(area)
    if truncation >= 3:
        curvature = np.zeros(d)
        for dim in range(d):
            second_diff = np.diff(increments[:, dim])
            curvature[dim] = np.sum(np.abs(second_diff))
        features.append(curvature)
    return np.concatenate([f.flatten() for f in features])


# ============================================================================
# MMD COMPUTATION (Definition 2.20, Eq. 29)
# ============================================================================

def compute_mmd(ensemble_x: List[np.ndarray],
                 ensemble_y: List[np.ndarray],
                 truncation: int = 2,
                 rbf_sigma: Optional[float] = None) -> float:
    n, m = len(ensemble_x), len(ensemble_y)
    if n == 0 or m == 0:
        return 0.0

    sigs_x = np.array([compute_signature_features(p, truncation) for p in ensemble_x])
    sigs_y = np.array([compute_signature_features(p, truncation) for p in ensemble_y])

    if rbf_sigma is not None:
        diff_xx = sigs_x[:, None] - sigs_x[None, :]
        K_xx = np.exp(-np.sum(diff_xx ** 2, axis=2) / (2 * rbf_sigma ** 2))
        diff_yy = sigs_y[:, None] - sigs_y[None, :]
        K_yy = np.exp(-np.sum(diff_yy ** 2, axis=2) / (2 * rbf_sigma ** 2))
        diff_xy = sigs_x[:, None] - sigs_y[None, :]
        K_xy = np.exp(-np.sum(diff_xy ** 2, axis=2) / (2 * rbf_sigma ** 2))
    else:
        K_xx = sigs_x @ sigs_x.T
        K_yy = sigs_y @ sigs_y.T
        K_xy = sigs_x @ sigs_y.T

    mmd2 = ((np.sum(K_xx) - np.trace(K_xx)) / max(n * (n - 1), 1)
            - 2 * np.sum(K_xy) / (n * m)
            + (np.sum(K_yy) - np.trace(K_yy)) / max(m * (m - 1), 1))
    return np.sqrt(max(0.0, mmd2))


# ============================================================================
# SUB-PATH & ENSEMBLE CONSTRUCTION (Definitions 3.1–3.2)
# ============================================================================

def extract_subpaths(path: np.ndarray, h1: int) -> List[np.ndarray]:
    n = len(path)
    n_subpaths = n // h1
    return [path[j * h1:(j + 1) * h1].copy() for j in range(n_subpaths)]


def build_ensembles(subpaths: List[np.ndarray], h2: int) -> List[List[np.ndarray]]:
    n = len(subpaths)
    if n < h2:
        return [subpaths[:]]
    return [subpaths[k:k + h2] for k in range(n - h2 + 1)]


# ============================================================================
# AUTO-EVALUATOR (Sec 3.2)
# ============================================================================

@dataclass
class DetectionConfig:
    h1: int = H1
    h2: int = H2
    transforms: list = None
    truncation: int = TRUNCATION
    rbf_sigma: float = RBF_SIGMA
    lags: list = None
    memory_window: int = MEMORY_WINDOW
    alpha: float = ALPHA

    def __post_init__(self):
        if self.transforms is None:
            self.transforms = ['time', 'state']
        if self.lags is None:
            self.lags = [1, 2]


class RegimeDetector:
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        np.random.seed(SEED)

    def compute_lag_score(self, ensembles, current_idx):
        if current_idx < max(self.config.lags):
            return 0.0
        scores = []
        for lag in self.config.lags:
            ref_idx = current_idx - lag
            if 0 <= ref_idx < len(ensembles) and 0 <= current_idx < len(ensembles):
                mmd = compute_mmd(
                    ensembles[ref_idx], ensembles[current_idx],
                    self.config.truncation, self.config.rbf_sigma
                )
                scores.append(mmd)
        return np.mean(scores) if scores else 0.0

    def fit_threshold(self, scores, alpha=0.05):
        if len(scores) < 5:
            return np.percentile(scores, 100 * (1 - alpha)) if scores else 1e-6
        s = np.array(scores)
        mean, var = np.mean(s), np.var(s)
        if var < 1e-10 or mean < 1e-10:
            return np.percentile(s, 100 * (1 - alpha))
        shape = mean ** 2 / var
        scale = var / mean
        try:
            return stats.gamma.ppf(1 - alpha, shape, scale=scale)
        except Exception:
            return np.percentile(s, 100 * (1 - alpha))

    def detect_regime_changes(self, path: np.ndarray) -> Dict:
        transformed = apply_transforms(path, self.config.transforms)
        subpaths = extract_subpaths(transformed, self.config.h1)
        ensembles = build_ensembles(subpaths, self.config.h2)
        n_ensembles = len(ensembles)

        print(f"  Computing MMD scores for {n_ensembles} ensembles...")
        scores = []
        for i in range(n_ensembles):
            score = self.compute_lag_score(ensembles, i)
            scores.append(score)
            if i % 50 == 0:
                print(f"    Progress: {i}/{n_ensembles}")

        scores = np.array(scores)

        # Dynamic thresholds (Gamma fit, Sec 3.2)
        thresholds = np.zeros(n_ensembles)
        for i in range(n_ensembles):
            w_start = max(0, i - self.config.memory_window)
            window_scores = scores[w_start:i + 1].tolist()
            thresholds[i] = self.fit_threshold(window_scores, self.config.alpha)

        change_mask = scores > thresholds
        change_points = np.where(change_mask)[0]

        # Safety checks
        n_changes = len(change_points)
        if n_changes == 0:
            print("  [WARN] Zero changes; using top-percentile fallback")
            fallback_threshold = np.percentile(scores, 95)
            change_mask = scores > fallback_threshold
            change_points = np.where(change_mask)[0]
        elif n_changes > n_ensembles * 0.30:
            print(f"  [WARN] {n_changes} changes too many; tightening to 99th pct")
            fallback_threshold = np.percentile(scores, 99)
            change_mask = scores > fallback_threshold
            change_points = np.where(change_mask)[0]

        return {
            'scores': scores,
            'thresholds': thresholds,
            'change_points': change_points,
            'change_mask': change_mask,
            'n_subpaths': len(subpaths),
            'n_ensembles': n_ensembles,
        }


# ============================================================================
# CHANGE-POINT → DAILY REGIME MAPPING
# ============================================================================

def changepoints_to_daily_labels(change_points, n_ensembles, h1, h2, n_days):
    """Map ensemble-level change points to per-day regime labels."""
    regimes = np.zeros(n_days, dtype=int)
    current_regime = 0
    change_days = sorted(cp * h1 for cp in change_points if cp * h1 < n_days)

    for d in range(n_days):
        while change_days and d >= change_days[0]:
            current_regime += 1
            change_days.pop(0)
        regimes[d] = current_regime

    # Make contiguous 0-based
    unique = np.unique(regimes)
    rmap = {old: new for new, old in enumerate(sorted(unique))}
    regimes = np.array([rmap[r] for r in regimes])
    return regimes


# ============================================================================
# POST-PROCESSING: MERGE SHORT SEGMENTS + RE-CLUSTER
# ============================================================================

def merge_short_segments(regimes, min_length=MIN_SEGMENT_DAYS):
    """Merge segments shorter than min_length into their neighbor."""
    merged = regimes.copy()
    changed = True
    while changed:
        changed = False
        # Find segment boundaries
        segments = []
        start = 0
        for i in range(1, len(merged)):
            if merged[i] != merged[i - 1]:
                segments.append((start, i))
                start = i
        segments.append((start, len(merged)))

        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            if seg_len < min_length:
                # Merge into the neighbor with the most similar regime
                # Prefer the longer neighbor
                left_regime = merged[seg_start - 1] if seg_start > 0 else None
                right_regime = merged[seg_end] if seg_end < len(merged) else None

                # Count lengths of each regime
                regime_lengths = {}
                for s_start, s_end in segments:
                    r = merged[s_start]
                    regime_lengths[r] = regime_lengths.get(r, 0) + (s_end - s_start)

                if left_regime is not None and right_regime is not None:
                    # Merge into whichever neighbor regime has longer total length
                    target = left_regime if regime_lengths.get(left_regime, 0) >= regime_lengths.get(right_regime, 0) else right_regime
                elif left_regime is not None:
                    target = left_regime
                elif right_regime is not None:
                    target = right_regime
                else:
                    continue

                merged[seg_start:seg_end] = target
                changed = True
                break  # restart

    # Renumber contiguously
    unique = np.unique(merged)
    rmap = {old: new for new, old in enumerate(sorted(unique))}
    merged = np.array([rmap[r] for r in merged])
    return merged


def cluster_regimes_by_features(regimes, prices_series, n_clusters=TARGET_N_REGIMES):
    """
    Re-cluster regimes based on their return/vol profile using K-means.

    This maps many change-point-induced regime IDs into fewer meaningful groups.
    """
    from sklearn.cluster import KMeans

    # Compute per-regime feature profiles
    returns = prices_series.pct_change().fillna(0).values
    # Ensure same length
    n = min(len(regimes), len(returns))
    regimes = regimes[:n]
    returns = returns[:n]

    # Features: mean daily return + mean daily volatility per regime
    unique_regimes = np.unique(regimes)
    regime_features = {}

    for r in unique_regimes:
        mask = regimes == r
        if mask.sum() < 2:
            regime_features[r] = np.array([0.0, 0.01])
            continue
        r_vals = returns[mask]
        mean_ret = np.mean(r_vals)
        std_ret = np.std(r_vals)
        regime_features[r] = np.array([mean_ret, std_ret])

    # Build feature matrix
    X = np.array([regime_features[r] for r in sorted(unique_regimes)])

    # Handle degenerate case
    if len(X) <= n_clusters:
        # Already fewer regimes than target
        return regimes, regimes  # identity mapping

    # K-means clustering
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    km.fit(X)
    original_to_cluster = {old: int(km.labels_[i]) for i, old in enumerate(sorted(unique_regimes))}

    # Map labels
    new_regimes = np.array([original_to_cluster[r] for r in regimes])

    # Ensure contiguous 0-based
    unique_new = np.unique(new_regimes)
    rmap = {old: new for new, old in enumerate(sorted(unique_new))}
    new_regimes = np.array([rmap[r] for r in new_regimes])

    return new_regimes, original_to_cluster


# ============================================================================
# VOLATILITY-BASED FALLBACK
# ============================================================================

def volatility_fallback(prices_eval, n_regimes=3):
    """
    Volatility-based regime detection fallback.
    3 regimes: low-vol (bull), medium-vol (sideways), high-vol (bear/crisis)
    """
    returns = prices_eval.pct_change().dropna()
    rolling_vol = returns.rolling(20).std() * np.sqrt(252)
    rolling_vol = rolling_vol.fillna(rolling_vol.median())

    vol_values = rolling_vol.values

    # Quantile-based breakpoints
    breakpoints = np.quantile(vol_values, [0, 1.0 / n_regimes, 2.0 / n_regimes, 1.0])

    labels = np.zeros(len(vol_values), dtype=int)
    for i in range(n_regimes):
        low = breakpoints[i]
        high = breakpoints[i + 1]
        if i < n_regimes - 1:
            mask = (vol_values >= low) & (vol_values < high)
        else:
            mask = (vol_values >= low) & (vol_values <= high)
        labels[mask] = i

    # Contiguous 0-based
    unique = np.unique(labels)
    rmap = {old: new for new, old in enumerate(sorted(unique))}
    labels = np.array([rmap[l] for l in labels])

    transitions = np.zeros(len(labels), dtype=int)
    for i in range(1, len(labels)):
        transitions[i] = 1 if labels[i] != labels[i - 1] else 0

    confidences = np.ones(len(labels), dtype=float)
    for i in range(len(labels)):
        vol = vol_values[i]
        low_bp = breakpoints[labels[i]]
        high_bp = breakpoints[min(labels[i] + 1, n_regimes)]
        rng = high_bp - low_bp
        if rng > 0:
            dist_from_boundary = min(abs(vol - low_bp), abs(vol - high_bp))
            confidences[i] = min(1.0, 0.5 + 0.5 * dist_from_boundary / (rng / 2))
        else:
            confidences[i] = 0.5

    return labels, transitions, confidences


# ============================================================================
# REGIME LABELING
# ============================================================================

def assign_regime_labels(regimes, prices_eval):
    """Assign human-readable labels based on return/vol profile of each regime.
    
    Uses a classification matrix to avoid duplicate labels:
      high vol + neg return → crisis
      high vol + pos return → volatile
      med vol + neg return → bear
      med vol + pos return → sideways
      low vol + pos return → bull
      low vol + ~0 return  → calm
    """
    returns = prices_eval.pct_change().fillna(0).values
    n = min(len(regimes), len(returns))

    # Build (mean_return, vol) profile per regime
    profiles = {}
    for r in sorted(np.unique(regimes)):
        mask = regimes[:n] == r
        if mask.sum() < 2:
            profiles[r] = (0.0, 0.01)
            continue
        mean_ret = np.mean(returns[:n][mask]) * 100
        std_ret = np.std(returns[:n][mask]) * 100
        profiles[r] = (mean_ret, std_ret)

    # Sort regimes by vol so we can assign unique labels
    sorted_by_vol = sorted(profiles.keys(), key=lambda r: profiles[r][1])

    # Assign labels ensuring uniqueness: lowest vol → bull/calm, highest vol → crisis
    n_regimes = len(sorted_by_vol)
    label_map = {}

    if n_regimes == 1:
        label_map[sorted_by_vol[0]] = "neutral"
    elif n_regimes == 2:
        # Low vol = bull, high vol = bear
        label_map[sorted_by_vol[0]] = "bull"
        label_map[sorted_by_vol[1]] = "bear"
    elif n_regimes == 3:
        label_map[sorted_by_vol[0]] = "bull"
        label_map[sorted_by_vol[1]] = "sideways"
        label_map[sorted_by_vol[2]] = "bear"
    elif n_regimes >= 4:
        # Split extremes further
        # Lowest vol → bull, next → calm, ..., second-highest → sideways, highest → crisis
        labels_pool = ["bull", "calm", "sideways", "bear", "volatile", "crisis"]
        # Use volatility-ordered assignment
        for i, r in enumerate(sorted_by_vol):
            if i == 0:
                label_map[r] = "bull"
            elif i == n_regimes - 1:
                # Highest vol — check if returns are negative
                if profiles[r][0] < -0.01:
                    label_map[r] = "crisis"
                else:
                    label_map[r] = "volatile"
            elif i == n_regimes - 2:
                label_map[r] = "bear" if profiles[r][0] < 0 else "sideways"
            elif i == 1:
                label_map[r] = "calm"
            else:
                label_map[r] = "sideways"

    # Ensure no duplicate labels — if duplicates, add numeric suffix
    seen = {}
    for r in sorted(label_map.keys()):
        lbl = label_map[r]
        if lbl in seen.values():
            # Find the regime with shorter duration, rename it
            mask_a = regimes[:n] == r
            label_map[r] = f"{lbl}_{r}"
        seen[r] = label_map[r]

    return label_map


def compute_confidence(regimes, change_day_indices):
    """Compute per-day confidence: higher in middle of regime, lower near transitions."""
    n = len(regimes)
    conf = np.ones(n, dtype=float)

    # Find segment boundaries
    boundaries = [0] + list(change_day_indices) + [n]

    for seg_i in range(len(boundaries) - 1):
        seg_start = boundaries[seg_i]
        seg_end = boundaries[seg_i + 1]
        seg_len = seg_end - seg_start

        for d in range(seg_start, seg_end):
            pos_in_seg = d - seg_start
            if seg_len <= 1:
                conf[d] = 0.5
            else:
                dist_from_edge = min(pos_in_seg, seg_len - 1 - pos_in_seg)
                conf[d] = 0.5 + 0.5 * (dist_from_edge / (seg_len / 2))

    return np.clip(conf, 0.0, 1.0)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_output(df, metadata, eval_start, eval_end):
    """Validate output against the spec. Returns list of error messages."""
    errors = []

    # Required columns
    for col in ['date', 'predicted_regime', 'confidence', 'transition_flag']:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # Date range
    actual_start = str(df['date'].min())
    if actual_start > eval_start:
        errors.append(f"Start date {actual_start} after expected {eval_start}")

    # predicted_regime: integer, non-negative, contiguous 0-based
    if 'predicted_regime' in df.columns:
        if not pd.api.types.is_integer_dtype(df['predicted_regime']):
            # Check if they can be cast to int
            try:
                df['predicted_regime'] = df['predicted_regime'].astype(int)
            except:
                errors.append("predicted_regime cannot be cast to integer")
        if (df['predicted_regime'] < 0).any():
            errors.append("predicted_regime contains negative values")
        unique_regimes = sorted(df['predicted_regime'].unique())
        expected = list(range(len(unique_regimes)))
        if list(unique_regimes) != expected:
            errors.append(f"Regime IDs not contiguous 0-based: got {unique_regimes}")

    # confidence in [0, 1]
    if 'confidence' in df.columns:
        if (df['confidence'] < 0).any() or (df['confidence'] > 1).any():
            errors.append("confidence out of [0,1] range")

    # transition_flag binary, first = 0
    if 'transition_flag' in df.columns:
        if not df['transition_flag'].isin([0, 1]).all():
            errors.append("transition_flag contains non-binary values")
        if df['transition_flag'].iloc[0] != 0:
            errors.append("First row transition_flag must be 0")

    # No NaN in required columns
    for col in ['date', 'predicted_regime', 'confidence', 'transition_flag']:
        if col in df.columns and df[col].isnull().any():
            errors.append(f"NaN in: {col}")

    # metadata consistency
    if 'num_regimes' in metadata and 'predicted_regime' in df.columns:
        expected_n = df['predicted_regime'].max() + 1
        if metadata['num_regimes'] != expected_n:
            errors.append(f"metadata num_regimes={metadata['num_regimes']} != data {expected_n}")

    # regime_labels_map covers all regimes
    if 'regime_labels_map' in metadata and 'predicted_regime' in df.columns:
        for r in sorted(df['predicted_regime'].unique()):
            if str(r) not in metadata['regime_labels_map']:
                errors.append(f"Regime {r} missing from regime_labels_map")

    return errors


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("NON-PARAMETRIC ONLINE MARKET REGIME DETECTION (v2 REWRITE)")
    print("Implementation of arXiv:2306.15835 (Horvath & Issa, 2023)")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------------
    print("\n[1] Loading S&P 500 data...")
    df = pd.read_parquet(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    eval_start = pd.Timestamp(EVAL_START)
    eval_end = pd.Timestamp(EVAL_END)
    df_eval = df.loc[eval_start:eval_end].copy()
    n_eval = len(df_eval)
    print(f"  Total rows: {len(df)}, Eval window: {n_eval} days")
    print(f"  Date range: {df_eval.index.min().date()} to {df_eval.index.max().date()}")

    # ---------------------------------------------------------------
    # 2. BUILD CONTEXT PATH (1yr lookback + eval window)
    # ---------------------------------------------------------------
    print("\n[2] Building price path with lookback...")
    lookback_start = eval_start - pd.Timedelta(days=400)
    df_context = df.loc[lookback_start:eval_end].copy()

    prices = df_context['close'].values
    dates_context = df_context.index
    n_total = len(prices)

    # Time-augmented path
    t_normalized = np.arange(n_total) / max(1, n_total - 1)
    path = np.column_stack([t_normalized, prices])
    print(f"  Path shape: {path.shape} ({n_total} days with lookback)")

    # ---------------------------------------------------------------
    # 3. MMD REGIME DETECTION
    # ---------------------------------------------------------------
    print("\n[3] Running signature-kernel MMD regime detection...")
    config = DetectionConfig(
        h1=H1, h2=H2,
        transforms=['time', 'state', 'incr'],
        truncation=TRUNCATION,
        rbf_sigma=RBF_SIGMA,
        lags=LAGS,
        memory_window=MEMORY_WINDOW,
        alpha=ALPHA
    )
    detector = RegimeDetector(config)
    results = detector.detect_regime_changes(path)

    n_cp = len(results['change_points'])
    print(f"  MMD change points: {n_cp}")
    print(f"  MMD scores — Mean: {np.mean(results['scores']):.6f}, "
          f"Std: {np.std(results['scores']):.6f}, "
          f"Max: {np.max(results['scores']):.6f}")

    # ---------------------------------------------------------------
    # 4. MAP TO DAILY REGIMES + SLICE EVAL WINDOW
    # ---------------------------------------------------------------
    print("\n[4] Mapping to daily regimes in evaluation window...")
    regimes_all = changepoints_to_daily_labels(
        results['change_points'], results['n_ensembles'], config.h1, config.h2, n_total
    )

    # Slice eval window
    eval_start_idx = np.where(dates_context >= eval_start)[0][0]
    regimes_eval = regimes_all[eval_start_idx:]

    n_raw_regimes = len(np.unique(regimes_eval))
    print(f"  Raw regimes in eval window: {n_raw_regimes}")

    # ---------------------------------------------------------------
    # 5. POST-PROCESS: MERGE SHORT SEGMENTS + CLUSTER
    # ---------------------------------------------------------------
    print("\n[5] Post-processing: merge short segments, cluster into target regimes...")

    # Merge segments shorter than min duration
    regimes_merged = merge_short_segments(regimes_eval, min_length=MIN_SEGMENT_DAYS)
    n_merged = len(np.unique(regimes_merged))
    print(f"  After merging: {n_merged} regimes")

    # Check if we have meaningful variation; if not, use fallback
    if n_merged <= 1:
        print("  [WARN] Only 1 regime detected — switching to volatility fallback")
        prices_eval = df_eval['close']
        labels, trans, conf = volatility_fallback(prices_eval, n_regimes=3)
        n_regimes = 3
        label_map = {0: "bull", 1: "sideways", 2: "bear"}
        used_fallback = True
    elif n_merged > TARGET_N_REGIMES:
        # Too many regimes — cluster into target number
        print(f"  Clustering {n_merged} raw regimes into {TARGET_N_REGIMES} target regimes...")
        prices_eval = df_eval['close']
        regimes_clustered, cluster_map = cluster_regimes_by_features(
            regimes_merged, prices_eval, n_clusters=TARGET_N_REGIMES
        )
        labels = regimes_clustered
        n_regimes = len(np.unique(labels))
        label_map = assign_regime_labels(labels, prices_eval)
        used_fallback = False
    else:
        # Reasonable number — just relabel
        labels = regimes_merged
        n_regimes = n_merged
        prices_eval = df_eval['close']
        label_map = assign_regime_labels(labels, prices_eval)
        used_fallback = False

    # Compute transitions
    transitions = np.zeros(len(labels), dtype=int)
    for i in range(1, len(labels)):
        transitions[i] = 1 if labels[i] != labels[i - 1] else 0

    # Compute confidence
    change_days = np.where(transitions == 1)[0].tolist()
    conf = compute_confidence(labels, change_days)

    print(f"  Final: {n_regimes} regimes, {int(transitions.sum())} transitions "
          f"({100 * transitions.mean():.1f}% transition rate)")
    for r in sorted(np.unique(labels)):
        mask = labels == r
        print(f"    Regime {r} ({label_map[r]}): {mask.sum()} days")

    # ---------------------------------------------------------------
    # 6. BUILD OUTPUT DATAFRAME
    # ---------------------------------------------------------------
    print("\n[6] Building output CSV...")
    dates_out = df_eval.index[:len(labels)]
    n_out = len(labels)

    output_df = pd.DataFrame({
        'date': dates_out.strftime('%Y-%m-%d'),
        'predicted_regime': labels.astype(int),
        'confidence': np.round(conf[:n_out], 4),
        'transition_flag': transitions.astype(int),
        'regime_label': [label_map.get(int(l), f'regime_{l}') for l in labels],
    })

    # ---------------------------------------------------------------
    # 7. BUILD METADATA JSON
    # ---------------------------------------------------------------
    print("\n[7] Building metadata JSON...")
    regime_labels_map = {str(int(r)): label_map.get(int(r), f'regime_{r}')
                         for r in sorted(np.unique(labels))}

    detection_method = "volatility_fallback" if used_fallback else "sig_mmd_changepoint"
    metadata = {
        "algorithm": ALGO_NAME,
        "algorithm_full_name": "Signature-Kernel MMD Regime Detection (Horvath–Issa 2023)",
        "algorithm_family": "changepoint",
        "paper_reference": "arXiv:2306.15835",
        "num_regimes": int(n_regimes),
        "regime_labels_map": regime_labels_map,
        "parameters": {
            "h1": H1,
            "h2": H2,
            "truncation": TRUNCATION,
            "rbf_sigma": RBF_SIGMA,
            "lags": LAGS,
            "memory_window": MEMORY_WINDOW,
            "alpha": ALPHA,
            "transforms": ["time", "state", "incr"],
            "min_segment_days": MIN_SEGMENT_DAYS,
            "target_n_regimes": TARGET_N_REGIMES,
            "training_window_days": 400,
            "features_used": ["daily_return", "daily_vol_20d", "signature_features"],
            "detection_method": detection_method,
            "random_seed": SEED,
        },
        "training_window_days": 400,
        "features_used": ["daily_return", "daily_vol_20d", "signature_features"],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_spec_version": "1.0",
    }

    # ---------------------------------------------------------------
    # 8. VALIDATE
    # ---------------------------------------------------------------
    print("\n[8] Validating output...")
    errors = validate_output(output_df, metadata, EVAL_START, EVAL_END)
    if errors:
        print(f"  {len(errors)} validation issues found, attempting fixes...")
        # Auto-fix: clip confidence
        output_df['confidence'] = output_df['confidence'].clip(0.0, 1.0)
        # Auto-fix: first transition_flag = 0
        output_df.loc[0, 'transition_flag'] = 0
        # Re-validate
        errors2 = validate_output(output_df, metadata, EVAL_START, EVAL_END)
        if errors2:
            print(f"  {len(errors2)} remaining issues:")
            for e in errors2:
                print(f"    - {e}")
        else:
            print("  All fixes applied, validation now passes!")
    else:
        print("  All validation checks passed!")

    # Final stats
    print(f"\n  Output: {len(output_df)} rows, {len(output_df['predicted_regime'].unique())} regimes")
    print(f"  Date range: {output_df['date'].iloc[0]} to {output_df['date'].iloc[-1]}")
    print(f"  Transitions: {int(output_df['transition_flag'].sum())} "
          f"({100 * output_df['transition_flag'].mean():.1f}%)")

    # ---------------------------------------------------------------
    # 9. WRITE OUTPUT
    # ---------------------------------------------------------------
    print("\n[9] Writing output files...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, f"{ALGO_NAME}_regimes.csv")
    output_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  CSV: {csv_path}")

    json_path = os.path.join(OUTPUT_DIR, f"{ALGO_NAME}_metadata.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  JSON: {json_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Algorithm: {metadata['algorithm_full_name']}")
    print(f"  Paper: {metadata['paper_reference']}")
    print(f"  Window: {output_df['date'].iloc[0]} – {output_df['date'].iloc[-1]}")
    print(f"  Days: {len(output_df)}, Regimes: {metadata['num_regimes']}")
    print(f"  Labels: {regime_labels_map}")
    for r in sorted(np.unique(labels)):
        mask = labels == r
        rets = prices_eval.pct_change().fillna(0).values[:len(mask)]
        if mask.sum() > 0:
            mr = np.mean(rets[mask]) * 100
            sd = np.std(rets[mask]) * 100
            print(f"    {label_map[int(r)]} (regime {r}): {mask.sum()} days, "
                  f"mean_ret={mr:+.3f}%/d, vol={sd:.3f}%/d")
    print("=" * 70)
    print("DONE")

    return output_df, metadata


if __name__ == "__main__":
    output_df, metadata = main()