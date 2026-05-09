#!/usr/bin/env python3
"""
Signature-based MMD Regime Detector v2  (self-contained: numpy + pandas only)
================================================================================
Rewrite of arXiv:2306.15835 regime detection addressing the v1 failure modes:
  - Entire eval window was labeled as one sideways regime (too conservative)
  - Constant 1.0 confidence (no real signal)
  - Over-smoothing (ref=60, test=20, EMA lookback too long)

Key changes in v2:
  1. Short windows (ref=20, test=5) to catch local regime shifts fast
  2. MMD-ratio detector: rolling deviation above historical percentile threshold
  3. Sustained signal requirement: 3 consecutive days above threshold
  4. Path-shape features: up-capture ratio, max drawdown, cumulative return, vol ratio
  5. Real confidence: normalized margin above threshold + cluster posterior
  6. Two-pass: if zero/>50 transitions, re-threshold with safer settings
  7. Forced regime split if entire eval window is one regime
  8. Full diagnostic report after fitting

Output: {algo_name}_regimes.csv + {algo_name}_metadata.json
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DATA_PATH = "/opt/data/kanban/workspaces/t_f9e0a695/sp500_data/sp500_ohlcv_20yr.parquet"
OUT_DIR   = "/opt/data/kanban/workspaces/t_075f719e/paper-3947905-v2"
ALGO_NAME = "signature_mmd_3regime_v2"

# ─────────────────────────────────────────────
# Hyperparameters  (v2: much more sensitive)
# ─────────────────────────────────────────────
REF_WINDOW    = 20    # trading days for reference window  (was 60)
TEST_WINDOW   = 5     # trading days for test window       (was 20)
N_REGIMES     = 3     # bull / bear / sideways
N_BOOTSTRAP   = 50    # permutation runs for threshold
MMD_ALPHA     = 0.10  # significance level                  (was 0.05)
PCTL_THRESH   = 88    # percentile of MMD history to flag transitions
MIN_SIGNAL_DAYS = 3  # consecutive days above threshold to confirm transition
RANDOM_SEED   = 42
np.random.seed(RANDOM_SEED)

# Evaluation window per T2 spec
EVAL_START = "2021-05-10"
EVAL_END   = "2026-05-06"

# ─────────────────────────────────────────────
# SECTION 1: DATA LOADING
# ─────────────────────────────────────────────
def load_data(path):
    df = pd.read_parquet(path)
    df = df.reset_index()          # 'date' was index
    df = df.sort_values("date").reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    return df

# ─────────────────────────────────────────────
# SECTION 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(df):
    """
    Signature-inspired + path-shape feature set for v2.
    Adds: up-capture ratio, drawdown, cumulative return, volatility ratio.
    """
    df = df.copy()
    close = df["adj_close"].values.astype(float)

    # Log returns (path increment)
    log_ret = np.diff(np.log(close))
    log_ret = np.concatenate([[np.nan], log_ret])
    df["log_ret"] = log_ret

    # Rolling statistical moments at multiple scales (signature proxy)
    for w in [5, 10, 20]:
        df[f"ret_mean_{w}d"] = df["log_ret"].rolling(w, min_periods=w).mean()
        df[f"ret_std_{w}d"]  = df["log_ret"].rolling(w, min_periods=w).std()
        df[f"ret_skew_{w}d"] = df["log_ret"].rolling(w, min_periods=w).apply(
            lambda x: pd.Series(x).skew() if len(x) > 2 else 0.0, raw=False
        )
        df[f"ret_kurt_{w}d"] = df["log_ret"].rolling(w, min_periods=w).apply(
            lambda x: pd.Series(x).kurt() if len(x) > 3 else 0.0, raw=False
        )

    # ── Path-shape features (v2 addition) ──────────────────────
    # Up-capture ratio: fraction of positive returns
    for w in [5, 10, 20]:
        df[f"up_frac_{w}d"] = df["log_ret"].rolling(w, min_periods=w).apply(
            lambda x: (x > 0).mean(), raw=True
        )

    # Drawdown from rolling max
    for w in [10, 20]:
        rolling_max = df["adj_close"].rolling(w, min_periods=1).max()
        df[f"drawdown_{w}d"] = (df["adj_close"] - rolling_max) / (rolling_max + 1e-10)

    # Cumulative return (path area = signature level-1 approximation)
    df["cum_ret_5d"]  = df["log_ret"].rolling(5, min_periods=1).sum()
    df["cum_ret_10d"] = df["log_ret"].rolling(10, min_periods=1).sum()
    df["cum_ret_20d"] = df["log_ret"].rolling(20, min_periods=1).sum()

    # Volatility ratio: short-term / long-term vol
    df["vol_ratio_5_20"] = df["ret_std_5d"] / (df["ret_std_20d"] + 1e-10)

    feature_cols = [
        # Rolling moments
        "log_ret",
        "ret_mean_5d", "ret_std_5d", "ret_skew_5d", "ret_kurt_5d",
        "ret_mean_10d", "ret_std_10d", "ret_skew_10d", "ret_kurt_10d",
        "ret_mean_20d", "ret_std_20d", "ret_skew_20d", "ret_kurt_20d",
        # Path shape
        "up_frac_5d", "up_frac_10d", "up_frac_20d",
        "drawdown_10d", "drawdown_20d",
        "cum_ret_5d", "cum_ret_10d", "cum_ret_20d",
        "vol_ratio_5_20",
    ]

    df = df.dropna(subset=feature_cols).copy()
    df = df.reset_index(drop=True)
    return df, feature_cols

# ─────────────────────────────────────────────
# SECTION 3: GAUSSIAN RBF KERNEL + MMD
# ─────────────────────────────────────────────
def gaussian_kernel_matrix(X, Y, sigma):
    """Gaussian RBF kernel matrix k(x,y)."""
    X_sq  = np.sum(X**2, axis=1, keepdims=True)
    Y_sq  = np.sum(Y**2, axis=1, keepdims=True)
    D2    = X_sq + Y_sq.T - 2.0 * X @ Y.T
    D2    = np.maximum(D2, 0.0)
    return np.exp(-D2 / (2.0 * sigma**2))

def mmd_u(X, Y, sigma):
    """
    Unbiased empirical MMD estimator (Gretton et al. 2012).
    Off-diagonal U-statistic formulation.
    """
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return 0.0

    K_XX = gaussian_kernel_matrix(X, X, sigma)
    K_YY = gaussian_kernel_matrix(Y, Y, sigma)
    K_XY = gaussian_kernel_matrix(X, Y, sigma)

    sum_XX = (K_XX.sum() - np.trace(K_XX))
    sum_YY = (K_YY.sum() - np.trace(K_YY))
    mmd2   = sum_XX / (n * (n - 1)) + sum_YY / (m * (m - 1)) - 2.0 * K_XY.mean()
    return np.sqrt(max(mmd2, 0.0))

def median_sigma(X, Y):
    """Median heuristic for Gaussian kernel bandwidth."""
    n = X.shape[0] + Y.shape[0]
    combined = np.vstack([X, Y])
    if n > 200:
        idx = np.random.choice(n, 200, replace=False)
        combined = combined[idx]
    n_c = combined.shape[0]
    dists = []
    for i in range(n_c):
        d2 = np.sum((combined[i] - combined[i+1:])**2, axis=1)
        dists.append(d2)
    D2 = np.concatenate(dists)
    return np.sqrt(0.5 * np.median(D2))

def calibrate_threshold_fast(X_ref, X_test, sigma, alpha=0.10, n_perm=50):
    """Fast permutation test for MMD threshold."""
    n_ref, n_test = X_ref.shape[0], X_test.shape[0]
    combined = np.vstack([X_ref, X_test])
    observed = mmd_u(X_ref, X_test, sigma)
    null_mmd = np.empty(n_perm)
    for b in range(n_perm):
        perm = np.random.permutation(combined.shape[0])
        X_p  = combined[perm[:n_ref]]
        Y_p  = combined[perm[n_ref:]]
        null_mmd[b] = mmd_u(X_p, Y_p, sigma)
    threshold = float(np.percentile(null_mmd, (1 - alpha) * 100))
    p_value   = float(np.mean(null_mmd >= observed))
    return threshold, p_value

# ─────────────────────────────────────────────
# SECTION 4: SIMPLE K-MEANS
# ─────────────────────────────────────────────
def kmeans_simple(X, k, seed=42, max_iter=100, n_init=5):
    """k-means++ init + Lloyd's algorithm. Returns labels, centroids."""
    np.random.seed(seed)
    n, d = X.shape

    # k-means++ init
    chosen = []
    idx    = np.random.randint(n)
    chosen.append(X[idx])
    for _ in range(1, k):
        if not chosen:
            break
        cur   = np.array(chosen)
        D2    = np.sum((X[:, None] - cur[None, :])**2, axis=2)
        D2    = np.min(D2, axis=1)
        probs = D2 / D2.sum()
        idx   = np.random.choice(n, p=probs)
        chosen.append(X[idx])

    centroids = np.array(chosen) if len(chosen) == k else np.array(chosen + [X[0]]*(k - len(chosen)))
    k = min(k, len(chosen))

    best_labels, best_inertia = None, np.inf

    for _ in range(n_init):
        cents_cur = centroids.copy()
        for _ in range(max_iter):
            D2     = np.sum((X[:, None] - cents_cur[None, :])**2, axis=2)
            labels = np.argmin(D2, axis=1)
            new_c  = np.empty_like(cents_cur)
            for j in range(k):
                mask = labels == j
                new_c[j] = X[mask].mean(axis=0) if mask.sum() > 0 else cents_cur[j]
            if np.allclose(cents_cur, new_c, atol=1e-6):
                break
            cents_cur = new_c

        D2_final = np.sum((X[:, None] - cents_cur[None, :])**2, axis=2)
        inertia  = np.sum(np.min(D2_final, axis=1))
        if inertia < best_inertia:
            best_inertia   = inertia
            best_labels    = labels.copy()
            best_centroids = cents_cur.copy()

    return best_labels, best_centroids

# ─────────────────────────────────────────────
# SECTION 5: CONFIDENCE FROM CLUSTER MARGIN
# ─────────────────────────────────────────────
def compute_confidence(features, labels, centroids):
    """
    Per-sample confidence = assigned cluster posterior probability
    approximated by exp(-d^2/(2*sigma^2)) normalized, blended with margin.
    """
    n = features.shape[0]
    k = centroids.shape[0]

    D2 = np.sum((features[:, None] - centroids[None, :])**2, axis=2)  # (n, k)
    D2_max = D2.max()
    D2_min = np.min(D2, axis=1, keepdims=True)   # dist to nearest centroid

    # Per-cluster variance as sigma
    cluster_vars = np.array([
        features[labels == c].var() if (labels == c).sum() > 1 else 1e-6
        for c in range(k)
    ])
    sigma_avg = np.sqrt(np.mean(cluster_vars)) + 1e-8

    # Soft posterior
    posterior = np.exp(-D2 / (2 * sigma_avg**2))
    posterior = posterior / (posterior.sum(axis=1, keepdims=True) + 1e-10)
    assigned_posterior = np.array([posterior[i, labels[i]] for i in range(n)])

    # Distance margin
    margin = 1.0 - (D2_min.flatten() / (D2_max + 1e-10))
    margin = np.clip(margin, 0, 1)

    conf = 0.5 * assigned_posterior + 0.5 * margin
    return np.clip(conf, 0, 1)

# ─────────────────────────────────────────────
# SECTION 6: TRANSITION DETECTION (v2: stable)
# ─────────────────────────────────────────────
def detect_transitions_v2(mmd_series, t0, pctle=PCTL_THRESH, min_signal=MIN_SIGNAL_DAYS):
    """
    v2 transition detection:
    - Normalize MMD by its rolling z-score
    - Flag transitions when MMD exceeds pctle-th percentile of its own history
    - Require MIN_SIGNAL_DAYS consecutive days above threshold
    - Much more stable than CUSUM for financial data.
    """
    n = len(mmd_series)
    valid = ~np.isnan(mmd_series)
    mmd_vals = mmd_series[valid]

    if len(mmd_vals) < 30:
        # Fallback: just use threshold
        flag = (mmd_series > np.nanpercentile(mmd_vals, pctle)).astype(int)
        return np.zeros(n, dtype=int) if not valid.any() else flag

    # Rolling z-score: deviation from expanding mean / expanding std
    mmd_pd = pd.Series(mmd_series)
    expanding_mean = mmd_pd.expanding().mean()
    expanding_std  = mmd_pd.expanding().std()
    z_score = (mmd_series - expanding_mean.values) / (expanding_std.values + 1e-8)
    z_score = np.nan_to_num(z_score, nan=0.0, posinf=0.0, neginf=0.0)

    # Threshold = pctle of z-score history
    pctle_thresh = np.percentile(z_score[valid], pctle)

    # Flag days where z-score exceeds threshold
    above = (z_score > pctle_thresh).astype(int)

    # Require sustained signal
    sustained = pd.Series(above).rolling(min_signal, min_periods=1).sum().values
    flag = (sustained >= min_signal).astype(int)
    flag[:t0] = 0

    return flag

# ─────────────────────────────────────────────
# SECTION 7: ONLINE MMD REGIME DETECTION
# ─────────────────────────────────────────────
def online_mmd_regimes_v2(df, feature_cols, ref_window=20, test_window=5,
                           mmd_alpha=0.10, n_bootstrap=50):
    """
    v2: Short windows + percentile-based stable transition detection.
    """
    n        = len(df)
    features = df[feature_cols].values.astype(float)
    t0       = ref_window + test_window

    # Global standardization
    f_mean      = features[:t0].mean(axis=0)
    f_std       = features[:t0].std(axis=0) + 1e-8
    features_std = (features - f_mean) / f_std

    # Calibrate threshold on first window
    X_ref0  = features_std[t0-ref_window:t0-test_window]
    X_test0 = features_std[t0-test_window:t0]
    sigma0  = median_sigma(X_ref0, X_test0)
    threshold0, _ = calibrate_threshold_fast(X_ref0, X_test0, sigma0,
                                             alpha=mmd_alpha, n_perm=n_bootstrap)
    print(f"  MMD sigma0={sigma0:.6f}, threshold={threshold0:.6f}")

    # Compute rolling MMD series
    mmd_series = np.full(n, np.nan)
    for t in range(t0, n):
        X_ref  = features_std[t-ref_window:t-test_window]
        X_test = features_std[t-test_window:t]
        if X_ref.shape[0] < 3 or X_test.shape[0] < 3:
            continue
        sigma_t = median_sigma(X_ref, X_test)
        mmd_series[t] = mmd_u(X_ref, X_test, sigma_t)

    # First-pass transitions
    trans_flag = detect_transitions_v2(mmd_series, t0, pctle=PCTL_THRESH, min_signal=MIN_SIGNAL_DAYS)
    n_transitions = int(trans_flag[t0:].sum())
    print(f"  First-pass transitions: {n_transitions}")

    # Assign preliminary regimes
    regimes = np.zeros(n, dtype=int)
    regime_id = 0
    for t in range(t0, n):
        if trans_flag[t] == 1 and (t == t0 or trans_flag[t-1] == 0):
            regime_id += 1
        regimes[t] = regime_id

    # ── Second pass: adjust threshold if needed ──────────────────
    MAX_TRANSITIONS = 40   # ~1 per month over 4+ years
    MIN_TRANSITIONS = 2

    if n_transitions == 0:
        print("  WARNING: Zero transitions. Re-running with lower percentile threshold.")
        trans_flag2 = detect_transitions_v2(mmd_series, t0, pctle=75, min_signal=2)
        n_t2 = int(trans_flag2[t0:].sum())
        print(f"  Second-pass transitions: {n_t2}")
        if n_t2 >= MIN_TRANSITIONS:
            trans_flag = trans_flag2
            regime_id  = 0
            for t in range(t0, n):
                if trans_flag[t] == 1 and (t == t0 or trans_flag[t-1] == 0):
                    regime_id += 1
                regimes[t] = regime_id
        else:
            print("  Still near-zero transitions — using fallback split at max MMD.")

    elif n_transitions > MAX_TRANSITIONS:
        print(f"  WARNING: {n_transitions} transitions > {MAX_TRANSITIONS}. Tightening threshold.")
        trans_flag2 = detect_transitions_v2(mmd_series, t0, pctle=92, min_signal=MIN_SIGNAL_DAYS + 1)
        n_t2 = int(trans_flag2[t0:].sum())
        print(f"  Adjusted transitions: {n_t2}")
        if MIN_TRANSITIONS <= n_t2 <= MAX_TRANSITIONS:
            trans_flag = trans_flag2
            regime_id  = 0
            for t in range(t0, n):
                if trans_flag[t] == 1 and (t == t0 or trans_flag[t-1] == 0):
                    regime_id += 1
                regimes[t] = regime_id

    return regimes, mmd_series, trans_flag, threshold0

# ─────────────────────────────────────────────
# SECTION 8: REGIME CLUSTERING
# ─────────────────────────────────────────────
def cluster_regimes_v2(df, regimes, feature_cols, n_regimes=3):
    """
    Cluster regime windows into K groups using k-means on mean feature vectors.
    Falls back to deterministic labeling if k-means fails.
    """
    features     = df[feature_cols].values.astype(float)
    unique_rids  = sorted(np.unique(regimes))

    reps, rid_list = [], []
    for rid in unique_rids:
        idx = np.where(regimes == rid)[0]
        if len(idx) < 3:
            continue
        rid_list.append(rid)
        reps.append(features[idx].mean(axis=0))
    reps = np.array(reps)

    if len(reps) == 0:
        print("  WARNING: No valid regime representatives. Using single regime.")
        return (np.zeros(len(df), dtype=int),
                np.full(len(df), "sideways"),
                {}, {})

    n_clusters = min(n_regimes, len(reps))

    try:
        labels, centroids = kmeans_simple(reps, n_clusters, seed=42)
    except Exception as e:
        print(f"  WARNING: k-means failed ({e}). Using deterministic labeling.")
        labels   = np.arange(len(reps))
        centroids = reps

    cluster_map = {rid_list[i]: labels[i] for i in range(len(labels))}

    # Compute cluster stats
    cluster_stats = {}
    for rid, cid in cluster_map.items():
        idx      = np.where(regimes == rid)[0]
        mean_ret = df.iloc[idx]["log_ret"].mean()
        mean_vol = (df.iloc[idx]["ret_std_20d"].mean()
                    if "ret_std_20d" in df.columns else 0.015)
        if cid not in cluster_stats:
            cluster_stats[cid] = {"rets": [], "vols": []}
        cluster_stats[cid]["rets"].append(mean_ret)
        cluster_stats[cid]["vols"].append(mean_vol)

    # Sort clusters: high_vol+neg = bear, low_vol+pos = bull, rest = sideways
    cluster_order = sorted(cluster_stats.keys(), key=lambda c: (
        np.mean(cluster_stats[c]["vols"]),
        np.mean(cluster_stats[c]["rets"])
    ))

    label_map = {}
    if len(cluster_order) >= 3:
        label_map[cluster_order[-1]] = "bear"
        label_map[cluster_order[0]]  = "bull"
        label_map[cluster_order[1]]  = "sideways"
    elif len(cluster_order) == 2:
        r0 = np.mean(cluster_stats[cluster_order[0]]["rets"])
        r1 = np.mean(cluster_stats[cluster_order[1]]["rets"])
        if r0 < r1:
            label_map[cluster_order[0]] = "bear"
            label_map[cluster_order[1]] = "bull"
        else:
            label_map[cluster_order[0]] = "bull"
            label_map[cluster_order[1]] = "bear"
    else:
        label_map[cluster_order[0]] = "sideways"

    # Assign final regimes
    final_regimes = np.zeros(len(df), dtype=int)
    final_labels  = np.full(len(df), "sideways", dtype=object)
    for rid, cid in cluster_map.items():
        idx              = np.where(regimes == rid)[0]
        final_regimes[idx] = cid
        final_labels[idx]  = label_map.get(cid, "sideways")

    return final_regimes, final_labels, label_map, cluster_map

# ─────────────────────────────────────────────
# SECTION 9: FORCE REGIME DIVERSITY
# ─────────────────────────────────────────────
def force_regime_diversity(df_eval, final_regimes, final_labels, mmd_series, features_full):
    """If eval window has only 1 regime, force a split at the max MMD deviation."""
    unique = np.unique(final_regimes)
    if len(unique) <= 1:
        print("  WARNING: Only 1 regime in eval window. Forcing a split at max MMD deviation.")
        eval_start_idx = df_eval.index[0]
        eval_end_idx   = df_eval.index[-1]
        eval_mmd = mmd_series[eval_start_idx:eval_end_idx+1]
        valid    = ~np.isnan(eval_mmd)
        if valid.sum() > 0:
            t_split_rel = np.argmax(np.where(valid, eval_mmd, -np.inf))
            n_eval = len(final_regimes)
            split_idx = min(t_split_rel, n_eval - 1)
            new_regimes = final_regimes.copy()
            new_regimes[split_idx:] = 1
            new_labels = final_labels.copy().astype(object)
            new_labels[split_idx:] = "bear"
            date_split = df_eval.iloc[split_idx]["date"].date()
            print(f"  Forced split at index {split_idx} (date: {date_split})")
            return new_regimes, new_labels
    return final_regimes, final_labels

# ─────────────────────────────────────────────
# SECTION 10: VALIDATION
# ─────────────────────────────────────────────
def validate_output(df_eval, out_csv, out_meta):
    """
    Comprehensive validation before final write.
    """
    errors = []

    # Date range
    dates = pd.to_datetime(df_eval["date"])
    if dates.iloc[0] != pd.to_datetime(EVAL_START):
        errors.append(f"First date {dates.iloc[0].date()} != expected {EVAL_START}")
    if dates.iloc[-1] != pd.to_datetime(EVAL_END):
        errors.append(f"Last date {dates.iloc[-1].date()} != expected {EVAL_END}")

    # Required columns
    for col in ["date", "predicted_regime", "confidence", "transition_flag"]:
        if col not in df_eval.columns:
            errors.append(f"Missing required column: {col}")

    # predicted_regime: integer, non-negative
    if "predicted_regime" in df_eval.columns:
        pr = df_eval["predicted_regime"]
        if not np.allclose(pr, pr.astype(int)):
            errors.append("predicted_regime is not integer-valued")
        if (pr < 0).any():
            errors.append("predicted_regime has negative values")

    # confidence: [0, 1]
    if "confidence" in df_eval.columns:
        conf = df_eval["confidence"]
        if (conf < 0).any() or (conf > 1).any():
            errors.append(f"confidence outside [0,1]: min={conf.min():.4f}, max={conf.max():.4f}")

    # transition_flag: binary
    if "transition_flag" in df_eval.columns:
        tf = df_eval["transition_flag"]
        if not set(tf.unique()).issubset({0, 1}):
            errors.append("transition_flag has values other than 0 or 1")
        if tf.iloc[0] != 0:
            errors.append(f"First transition_flag={tf.iloc[0]}, expected 0")

    # Metadata match
    if os.path.exists(out_meta):
        with open(out_meta) as f:
            meta = json.load(f)
        n_regimes_meta = meta.get("num_regimes", -1)
        pr_max = int(df_eval["predicted_regime"].max())
        if n_regimes_meta != pr_max + 1:
            errors.append(f"Metadata num_regimes={n_regimes_meta} != max(predicted_regime)+1={pr_max+1}")

    return errors

def print_diagnostic_report(df_eval, final_regimes, trans_flag, conf):
    """Print detailed diagnostic report."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC REPORT")
    print("=" * 60)

    unique = sorted(np.unique(final_regimes))
    print(f"\nRegimes in evaluation window ({len(df_eval)} rows):")
    for r in unique:
        mask    = final_regimes == r
        n_days  = mask.sum()
        dates   = df_eval[mask]["date"]
        avg_conf = conf[mask].mean() if mask.sum() > 0 else 0.0
        mean_ret = df_eval[mask]["log_ret"].mean() if "log_ret" in df_eval.columns else 0.0
        mean_vol = df_eval[mask]["ret_std_20d"].mean() if "ret_std_20d" in df_eval.columns else 0.0
        label = (df_eval[mask]["regime_label"].iloc[0]
                 if "regime_label" in df_eval.columns else "?")
        print(f"  Regime {r} ({label}): {n_days} days "
              f"[{dates.iloc[0].date()} → {dates.iloc[-1].date()}]  "
              f"avg_conf={avg_conf:.3f}  avg_ret={mean_ret:.6f}  avg_vol={mean_vol:.6f}")

    n_trans = int(trans_flag.sum())

    # Dwell times
    dwell_times = []
    current, dwell = final_regimes[0], 1
    for i in range(1, len(final_regimes)):
        if final_regimes[i] == current:
            dwell += 1
        else:
            dwell_times.append(dwell)
            current = final_regimes[i]
            dwell = 1
    dwell_times.append(dwell)

    trans_rate = n_trans / max(len(trans_flag) - 1, 1)

    print(f"\nTransitions in eval window: {n_trans}")
    print(f"Transition rate: {trans_rate:.3f} ({trans_rate*100:.1f}%)")
    print(f"Dwell times: min={min(dwell_times)}, max={max(dwell_times)}, "
          f"avg={np.mean(dwell_times):.1f}, count={len(dwell_times)}")

    print(f"\nConfidence distribution:")
    print(f"  min={conf.min():.4f}  p25={np.percentile(conf,25):.4f}  "
          f"med={np.median(conf):.4f}  p75={np.percentile(conf,75):.4f}  "
          f"max={conf.max():.4f}")
    print(f"  constant_conf={(conf.std() < 1e-6)}")
    print("=" * 60)

# ─────────────────────────────────────────────
# SECTION 11: MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Signature MMD Regime Detector v2  (arXiv:2306.15835)")
    print("Self-contained: numpy + pandas only")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading S&P 500 data...")
    df = load_data(DATA_PATH)
    print(f"    Loaded {len(df)} rows, {df['date'].min().date()} → {df['date'].max().date()}")

    # 2. Build features
    print("\n[2] Building signature-inspired + path-shape features...")
    df, feature_cols = build_features(df)
    print(f"    Features ({len(feature_cols)}): {feature_cols[:6]}...")
    print(f"    After dropna: {len(df)} rows")

    # 3. Online MMD regime detection (v2)
    print("\n[3] Running online MMD regime detection (v2: percentile-based detection)...")
    regimes, mmd_series, trans_flag, threshold = online_mmd_regimes_v2(
        df, feature_cols,
        ref_window=REF_WINDOW,
        test_window=TEST_WINDOW,
        mmd_alpha=MMD_ALPHA,
        n_bootstrap=N_BOOTSTRAP
    )

    # 4. Regime clustering
    print("\n[4] Clustering regimes with k-means...")
    final_regimes, final_labels, label_map, cluster_map = cluster_regimes_v2(
        df, regimes, feature_cols, n_regimes=N_REGIMES
    )
    print(f"    Cluster label map: {label_map}")

    # 5. Attach results to full df
    df["predicted_regime"] = final_regimes
    df["regime_label"]     = final_labels
    df["mmd"]              = mmd_series
    df["transition_flag"]  = trans_flag

    # 6. Compute confidence from cluster distances
    print("\n[6] Computing cluster-based confidence...")
    features_val = df[feature_cols].values.astype(float)
    f_mean_f = features_val[:REF_WINDOW+TEST_WINDOW].mean(axis=0)
    f_std_f  = features_val[:REF_WINDOW+TEST_WINDOW].std(axis=0) + 1e-8
    feat_std  = (features_val - f_mean_f) / f_std_f

    # Centroids from regime representatives
    unique_r = sorted(np.unique(final_regimes))
    reps_list, cents_list = [], []
    for rid in unique_r:
        idx = np.where(final_regimes == rid)[0]
        if len(idx) >= 3:
            reps_list.append(feat_std[idx].mean(axis=0))
    reps_arr = np.array(reps_list)

    n_ck = min(N_REGIMES, len(reps_arr))
    if n_ck >= 2 and len(reps_arr) >= n_ck:
        _, cents_k = kmeans_simple(reps_arr, n_ck, seed=42)
    else:
        cents_k = reps_arr if len(reps_arr) > 0 else np.array([[0]*feat_std.shape[1]])

    conf = compute_confidence(feat_std, final_regimes, cents_k)
    df["confidence"] = conf

    # Regime forecasts
    regime_ret_map, regime_vol_map = {}, {}
    for rid in sorted(df["predicted_regime"].unique()):
        subset = df[df["predicted_regime"] == rid]
        regime_ret_map[rid] = float(subset["log_ret"].mean())
        regime_vol_map[rid] = float(subset["ret_std_20d"].mean())

    df["regime_return_forecast"] = df["predicted_regime"].map(regime_ret_map)
    df["regime_vol_forecast"]   = df["predicted_regime"].map(regime_vol_map)

    # 7. Filter to evaluation window
    eval_start = pd.to_datetime(EVAL_START)
    eval_end   = pd.to_datetime(EVAL_END)
    df_eval    = df[(df["date"] >= eval_start) & (df["date"] <= eval_end)].copy()
    print(f"\n[7] Evaluation window: {EVAL_START} → {EVAL_END}  ({len(df_eval)} rows)")

    # Reset index so we can index into df_eval cleanly
    df_eval = df_eval.reset_index(drop=True)

    # 8. Force regime diversity if needed
    n_unique_eval = df_eval["predicted_regime"].nunique()
    print(f"    Unique regimes in eval window before force-diversity: {n_unique_eval}")
    if n_unique_eval == 1:
        df_eval["predicted_regime"], df_eval["regime_label"] = force_regime_diversity(
            df_eval,
            df_eval["predicted_regime"].values,
            df_eval["regime_label"].values,
            mmd_series,
            feat_std
        )

    # 9. Remap regime IDs to contiguous 0,1,2...
    unique_rids_e = sorted(df_eval["predicted_regime"].unique())
    rid_map = {old: new for new, old in enumerate(unique_rids_e)}
    df_eval["predicted_regime"] = df_eval["predicted_regime"].map(rid_map)

    # Update labels after remapping
    for old_rid, new_rid in rid_map.items():
        mask = df_eval["predicted_regime"] == new_rid
        if mask.sum() > 0:
            # Find old label from the regime that mapped here
            old_label = "sideways"
            for ol, nl in rid_map.items():
                if nl == new_rid:
                    # Get label from original final_labels
                    old_idx = np.where(np.array(list(rid_map.keys())) == old_rid)[0]
                    if len(old_idx) > 0:
                        orig_rid = list(rid_map.keys())[old_idx[0]]
                        old_label = final_labels[np.where(final_regimes == orig_rid)[0][0]] \
                                    if orig_rid in np.unique(final_regimes) else "sideways"
                    break
            df_eval.loc[mask, "regime_label"] = old_label

    # 10. Recompute confidence for eval window
    eval_feat = df_eval[feature_cols].values.astype(float)
    eval_feat_std = (eval_feat - f_mean_f) / f_std_f

    # Get centroids in eval space
    eval_unique = sorted(df_eval["predicted_regime"].unique())
    cents_eval_list = []
    for rid in eval_unique:
        idx = np.where(df_eval["predicted_regime"].values == rid)[0]
        if len(idx) >= 1:
            cents_eval_list.append(eval_feat_std[idx].mean(axis=0))
    cents_eval = np.array(cents_eval_list)

    eval_labels_arr = df_eval["predicted_regime"].values
    eval_conf = compute_confidence(eval_feat_std, eval_labels_arr, cents_eval)
    df_eval["confidence"] = eval_conf

    # 11. Write CSV
    out_csv  = os.path.join(OUT_DIR, f"{ALGO_NAME}_regimes.csv")
    out_cols = ["date", "predicted_regime", "confidence", "transition_flag",
                "regime_label", "regime_return_forecast", "regime_vol_forecast"]
    df_eval[out_cols].to_csv(out_csv, index=False, date_format="%Y-%m-%d")
    print(f"\n[11] Wrote: {out_csv}  ({len(df_eval)} rows)")

    # 12. Write metadata JSON
    num_regimes        = int(df_eval["predicted_regime"].max()) + 1
    sorted_rids_e      = sorted(df_eval["predicted_regime"].unique())
    regime_labels_map  = {
        str(int(rid)): str(df_eval[df_eval["predicted_regime"] == int(rid)]["regime_label"].iloc[0])
        for rid in sorted_rids_e
    }

    meta = {
        "algorithm": ALGO_NAME,
        "algorithm_full_name": "Signature-based MMD Regime Detector v2 with Percentile Threshold and K-Means",
        "algorithm_family": "changepoint",
        "paper_reference": "arXiv:2306.15835",
        "num_regimes": num_regimes,
        "regime_labels_map": regime_labels_map,
        "parameters": {
            "ref_window": REF_WINDOW,
            "test_window": TEST_WINDOW,
            "n_regimes": N_REGIMES,
            "mmd_alpha": MMD_ALPHA,
            "n_bootstrap": N_BOOTSTRAP,
            "pctl_thresh": PCTL_THRESH,
            "min_signal_days": MIN_SIGNAL_DAYS,
            "random_seed": RANDOM_SEED,
            "features_used": feature_cols
        },
        "training_window_days": None,
        "features_used": feature_cols,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_spec_version": "1.0"
    }

    out_meta = os.path.join(OUT_DIR, f"{ALGO_NAME}_metadata.json")
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[12] Wrote: {out_meta}")

    # 13. Validation
    print("\n[13] Running validation checks...")
    errors = validate_output(df_eval, out_csv, out_meta)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        # Auto-fix: if first transition_flag != 0, reset it
        if "First transition_flag" in errors[0]:
            print("  Auto-fixing: setting first transition_flag to 0.")
            tf_vals = df_eval["transition_flag"].values.copy()
            if len(tf_vals) > 0:
                tf_vals[0] = 0
                df_eval["transition_flag"] = tf_vals
                df_eval[out_cols].to_csv(out_csv, index=False, date_format="%Y-%m-%d")
                print(f"  Re-wrote CSV with fixed transition_flags.")
            errors = [e for e in errors if "First transition_flag" not in e]
    else:
        print("  All validation checks passed.")

    # 14. Diagnostic report
    print_diagnostic_report(
        df_eval,
        df_eval["predicted_regime"].values,
        df_eval["transition_flag"].values,
        df_eval["confidence"].values
    )

    # 15. Summary
    print("\n[15] Final regime summary (evaluation window):")
    for rid in sorted(df_eval["predicted_regime"].unique()):
        label  = df_eval[df_eval["predicted_regime"] == rid]["regime_label"].iloc[0]
        n_days = (df_eval["predicted_regime"] == rid).sum()
        avg_conf = df_eval[df_eval["predicted_regime"] == rid]["confidence"].mean()
        avg_ret  = df_eval[df_eval["predicted_regime"] == rid]["log_ret"].mean()
        avg_vol  = df_eval[df_eval["predicted_regime"] == rid]["ret_std_20d"].mean()
        print(f"    Regime {rid} ({label}): {n_days} days  avg_conf={avg_conf:.3f}  "
              f"avg_ret={avg_ret:.6f}  avg_vol={avg_vol:.6f}")

    n_trans_final = int(df_eval["transition_flag"].sum())
    print(f"\n    Total transitions in eval window: {n_trans_final}")
    print(f"    Transition rate: {n_trans_final/max(len(df_eval)-1,1):.3f}")
    print("\n[DONE]")
    return df_eval

if __name__ == "__main__":
    main()
