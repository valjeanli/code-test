#!/usr/bin/env python3
"""
Signature-based MMD Regime Detector  (self-contained: numpy + pandas only)
=======================================================================
Implementation based on arXiv:2306.15835 — "Non-parametric online market regime
detection and regime clustering for multidimensional and path-dependent data
structures" (Pauli et al., 2023).

Algorithm:
  1. Rolling signature-inspired features (log-returns, vol, skew, kurtosis)
  2. Sliding-window MMD two-sample test using Gaussian RBF kernel
  3. Online regime transition detection via permutation-calibrated threshold
  4. Simple k-means for regime clustering (K=3: bull/bear/sideways)

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
OUT_DIR   = "/opt/data/kanban/workspaces/t_16a54a09/paper-3947905"
ALGO_NAME = "signature_mmd_3regime"

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
REF_WINDOW    = 60    # trading days for reference window
TEST_WINDOW   = 20    # trading days for test window
N_REGIMES     = 3     # K for regime clustering
N_BOOTSTRAP   = 100   # permutation runs for threshold calibration
MMD_ALPHA     = 0.05  # significance level
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
# SECTION 2: FEATURE ENGINEERING (Signature-Inspired)
# ─────────────────────────────────────────────
def build_features(df):
    """
    Signature-inspired rolling features.
    Path signatures = iterated integrals of a path.
    Approximate with rolling statistical moments of log-returns.
    """
    df = df.copy()
    close = df["adj_close"].values.astype(float)

    # Log returns (path increment)
    log_ret = np.diff(np.log(close))
    log_ret = np.concatenate([[np.nan], log_ret])
    df["log_ret"] = log_ret

    # Rolling features — our "signature" vector at each time t
    for w in [5, 20, 60]:
        df[f"ret_mean_{w}d"] = df["log_ret"].rolling(w, min_periods=w).mean()
        df[f"ret_std_{w}d"] = df["log_ret"].rolling(w, min_periods=w).std()
        df[f"ret_skew_{w}d"] = df["log_ret"].rolling(w, min_periods=w).apply(
            lambda x: pd.Series(x).skew() if len(x) > 2 else 0.0, raw=False
        )
        df[f"ret_kurt_{w}d"] = df["log_ret"].rolling(w, min_periods=w).apply(
            lambda x: pd.Series(x).kurt() if len(x) > 3 else 0.0, raw=False
        )

    # Drawdown signature term
    rolling_max = df["adj_close"].rolling(20, min_periods=1).max()
    df["drawdown_20d"] = (df["adj_close"] - rolling_max) / (rolling_max + 1e-10)

    # Keep only rows with all features
    feature_cols = [
        "log_ret", "ret_mean_5d", "ret_std_5d", "ret_skew_5d", "ret_kurt_5d",
        "ret_mean_20d", "ret_std_20d", "ret_skew_20d", "ret_kurt_20d",
        "ret_mean_60d", "ret_std_60d", "ret_skew_60d", "ret_kurt_60d",
        "drawdown_20d"
    ]
    df = df.dropna(subset=feature_cols).copy()
    df = df.reset_index(drop=True)
    return df, feature_cols

# ─────────────────────────────────────────────
# SECTION 3: GAUSSIAN RBF KERNEL + MMD (numpy only)
# ─────────────────────────────────────────────
def gaussian_kernel_matrix(X, Y, sigma):
    """Gaussian RBF kernel matrix k(x,y) for all pairs (X rows, Y rows)."""
    # X: (n, d), Y: (m, d)
    X_sq = np.sum(X**2, axis=1, keepdims=True)    # (n, 1)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True)    # (m, 1)
    D2   = X_sq + Y_sq.T - 2.0 * X @ Y.T          # (n, m) pairwise sq distances
    D2   = np.maximum(D2, 0.0)                    # numerical floor
    return np.exp(-D2 / (2.0 * sigma**2))

def mmd_u(X, Y, sigma):
    """
    Unbiased empirical MMD estimator (Greitens et al. 2012).
    MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
    Uses U-statistic formulation to avoid diagonal terms.
    """
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        return 0.0

    K_XX = gaussian_kernel_matrix(X, X, sigma)
    K_YY = gaussian_kernel_matrix(Y, Y, sigma)
    K_XY = gaussian_kernel_matrix(X, Y, sigma)

    # Sum over off-diagonal entries only
    sum_XX = (K_XX.sum() - np.trace(K_XX))           # off-diag sum
    sum_YY = (K_YY.sum() - np.trace(K_YY))

    mmd2 = sum_XX / (n * (n - 1)) + sum_YY / (m * (m - 1)) - 2.0 * K_XY.mean()
    return np.sqrt(max(mmd2, 0.0))

def median_sigma(X, Y):
    """Median heuristic for Gaussian kernel bandwidth."""
    n = X.shape[0] + Y.shape[0]
    combined = np.vstack([X, Y])
    # Subsample for speed
    if n > 200:
        idx = np.random.choice(n, 200, replace=False)
        combined = combined[idx]
    # Pairwise squared distances (upper triangle only)
    n_c = combined.shape[0]
    dists = []
    for i in range(n_c):
        d2 = np.sum((combined[i] - combined[i+1:])**2, axis=1)
        dists.append(d2)
    D2 = np.concatenate(dists)
    return np.sqrt(0.5 * np.median(D2))

# ─────────────────────────────────────────────
# SECTION 4: PERMUTATION TEST FOR THRESHOLD
# ─────────────────────────────────────────────
def calibrate_threshold(X_ref, X_test, sigma, alpha=0.05, n_perm=100):
    """
    Permutation test: shuffle pooled data, split, compute MMD.
    Threshold = (1-alpha) quantile of null distribution.
    """
    n_ref, n_test = X_ref.shape[0], X_test.shape[0]
    combined = np.vstack([X_ref, X_test])
    observed = mmd_u(X_ref, X_test, sigma)

    null_mmd = np.empty(n_perm)
    for b in range(n_perm):
        perm = np.random.permutation(combined.shape[0])
        X_p = combined[perm[:n_ref]]
        Y_p = combined[perm[n_ref:]]
        null_mmd[b] = mmd_u(X_p, Y_p, sigma)

    threshold = float(np.percentile(null_mmd, (1 - alpha) * 100))
    p_value   = float(np.mean(null_mmd >= observed))
    return threshold, p_value

# ─────────────────────────────────────────────
# SECTION 5: SIMPLE K-MEANS (numpy only)
# ─────────────────────────────────────────────
def kmeans_simple(X, k, seed=42, max_iter=100, n_init=5):
    """
    Simple k-means++ initialization + Lloyd's algorithm.
    X: (n, d) float64
    Returns: labels (n,), centroids (k, d)
    """
    np.random.seed(seed)
    n, d = X.shape

    # k-means++ init
    chosen = []
    idx = np.random.randint(n)
    chosen.append(X[idx])
    for _ in range(1, k):
        if not chosen:
            break
        cur = np.array(chosen)                           # (m, d) m centroids so far
        D2  = np.sum((X[:, None] - cur[None, :])**2, axis=2)  # (n, m)
        D2  = np.min(D2, axis=1)                           # (n,)
        probs = D2 / D2.sum()
        idx = np.random.choice(n, p=probs)
        chosen.append(X[idx])

    centroids = np.array(chosen) if len(chosen) == k else np.array(chosen + [X[0]]*(k-len(chosen)))
    k = min(k, len(chosen))

    best_labels = None
    best_inertia = np.inf

    for _ in range(n_init):
        centroids_cur = centroids.copy()
        for _ in range(max_iter):
            # Assign
            D2 = np.sum((X[:, None] - centroids_cur[None, :])**2, axis=2)  # (n, k)
            labels = np.argmin(D2, axis=1)
            # Update
            new_cents = np.empty_like(centroids_cur)
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    new_cents[j] = X[mask].mean(axis=0)
                else:
                    new_cents[j] = centroids_cur[j]
            # Converged?
            if np.allclose(centroids_cur, new_cents, atol=1e-6):
                break
            centroids_cur = new_cents

        # Compute inertia and track best
        D2_final = np.sum((X[:, None] - centroids_cur[None, :])**2, axis=2)
        inertia  = np.sum(np.min(D2_final, axis=1))
        if inertia < best_inertia:
            best_inertia  = inertia
            best_labels   = labels.copy()
            best_centroids = centroids_cur.copy()

    return best_labels, best_centroids

# ─────────────────────────────────────────────
# SECTION 6: ONLINE MMD REGIME DETECTION
# ─────────────────────────────────────────────
def online_mmd_regimes(df, feature_cols, ref_window=60, test_window=20,
                       mmd_alpha=0.05, n_bootstrap=100):
    """
    Sliding-window MMD regime detection.
    Ref window = earlier window; Test window = most recent window.
    Transition detected when MMD exceeds threshold calibrated via permutation.
    Uses per-comparison standardization for scale-invariant kernel computation.
    """
    n = len(df)
    features = df[feature_cols].values.astype(float)
    t0 = ref_window + test_window

    # Global standardization for scale-invariant MMD
    f_mean = features[:t0].mean(axis=0)
    f_std  = features[:t0].std(axis=0) + 1e-8
    features_std = (features - f_mean) / f_std

    # Calibrate threshold on first window
    X_ref0  = features_std[t0-ref_window:t0-test_window]
    X_test0 = features_std[t0-test_window:t0]
    sigma0  = median_sigma(X_ref0, X_test0)
    threshold0, _ = calibrate_threshold(X_ref0, X_test0, sigma0,
                                          alpha=mmd_alpha, n_perm=n_bootstrap)
    print(f"  MMD sigma0={sigma0:.6f}, threshold={threshold0:.6f}")

    mmd_series = np.full(n, np.nan)
    trans_flag  = np.zeros(n, dtype=int)
    regimes     = np.zeros(n, dtype=int)

    # Rolling MMD EMA for adaptive baseline
    mmd_ema = 0.0
    ema_alpha = 0.1

    for t in range(t0, n):
        X_ref  = features_std[t-ref_window:t-test_window]
        X_test = features_std[t-test_window:t]
        if X_ref.shape[0] < 5 or X_test.shape[0] < 5:
            continue

        sigma_t = median_sigma(X_ref, X_test)
        mmd_t   = mmd_u(X_ref, X_test, sigma_t)
        mmd_series[t] = mmd_t

        # Adaptive EMA baseline
        if t == t0:
            mmd_ema = mmd_t
        else:
            mmd_ema = ema_alpha * mmd_t + (1 - ema_alpha) * mmd_ema

        # Transition: MMD significantly above its EMA baseline
        # Use ratio > 1.5 as transition signal
        if mmd_t > mmd_ema * 1.5 and mmd_t > threshold0:
            trans_flag[t] = 1

    # Require sustained signal: at least 2 of last 3 days above threshold
    roll_sum = pd.Series(trans_flag).rolling(3, min_periods=1).sum().values
    trans_flag_clean = (roll_sum >= 2).astype(int)
    trans_flag_clean[:t0] = 0

    # Assign regime IDs
    regime_id = 0
    for t in range(t0, n):
        if trans_flag_clean[t] == 1 and (t == t0 or trans_flag_clean[t-1] == 0):
            regime_id += 1
        regimes[t] = regime_id

    return regimes, mmd_series, trans_flag_clean, threshold0

# ─────────────────────────────────────────────
# SECTION 7: REGIME CLUSTERING
# ─────────────────────────────────────────────
def cluster_regimes(df, regimes, feature_cols, n_regimes=3):
    """
    Compute regime representatives and cluster into K groups
    using simple k-means on mean feature vectors per regime.
    """
    features = df[feature_cols].values.astype(float)
    unique_rids = sorted(np.unique(regimes))
    n_rids = len(unique_rids)

    # Compute representative features per regime
    reps = []
    for rid in unique_rids:
        idx = np.where(regimes == rid)[0]
        if len(idx) < 5:
            continue
        # Use mean feature vector of middle 60% of the regime window
        s, e = int(len(idx)*0.2), int(len(idx)*0.8)
        reps.append(features[idx[s:e]].mean(axis=0))
    reps = np.array(reps)

    if len(reps) < n_regimes:
        n_clusters = max(1, len(reps))
    else:
        n_clusters = n_regimes

    labels, centroids = kmeans_simple(reps, n_clusters, seed=42)

    # Map cluster IDs to regime IDs
    rid_list = [r for r in unique_rids if np.where(regimes == r)[0].size >= 5]
    cluster_map = {rid_list[i]: labels[i] for i in range(min(len(labels), len(rid_list)))}

    # Compute cluster stats (mean return + mean vol)
    cluster_stats = {}
    for rid, cid in cluster_map.items():
        idx = np.where(regimes == rid)[0]
        mean_ret = df.iloc[idx]["log_ret"].mean()
        mean_vol = df.iloc[idx]["ret_std_20d"].mean() if "ret_std_20d" in df.columns else 0.015
        if cid not in cluster_stats:
            cluster_stats[cid] = {"rets": [], "vols": []}
        cluster_stats[cid]["rets"].append(mean_ret)
        cluster_stats[cid]["vols"].append(mean_vol)

    # Sort clusters: high_vol+neg_ret=bear, low_vol+pos_ret=bull, rest=sideways
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
        # Sort by return: lower ret = bear
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

    # Final regime assignments
    final_regimes = np.zeros(len(df), dtype=int)
    final_labels  = np.full(len(df), "sideways", dtype=object)
    for rid, cid in cluster_map.items():
        idx = np.where(regimes == rid)[0]
        final_regimes[idx] = cid
        final_labels[idx]  = label_map.get(cid, "sideways")

    return final_regimes, final_labels, label_map, cluster_map

# ─────────────────────────────────────────────
# SECTION 8: MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Signature MMD Regime Detector  (arXiv:2306.15835)")
    print("Self-contained: numpy + pandas only")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading S&P 500 data...")
    df = load_data(DATA_PATH)
    print(f"    Loaded {len(df)} rows, {df['date'].min().date()} → {df['date'].max().date()}")

    # 2. Build features
    print("\n[2] Building signature-inspired features...")
    df, feature_cols = build_features(df)
    print(f"    Features ({len(feature_cols)}): {feature_cols[:6]}...")
    print(f"    After dropna: {len(df)} rows")

    # 3. Online MMD regime detection
    print("\n[3] Running online MMD regime detection...")
    regimes, mmd_series, trans_flag, threshold = online_mmd_regimes(
        df, feature_cols,
        ref_window=REF_WINDOW,
        test_window=TEST_WINDOW,
        mmd_alpha=MMD_ALPHA,
        n_bootstrap=N_BOOTSTRAP
    )

    # 4. Regime clustering
    print("\n[4] Clustering regimes with k-means...")
    final_regimes, final_labels, label_map, cluster_map = cluster_regimes(
        df, regimes, feature_cols, n_regimes=N_REGIMES
    )
    print(f"    Cluster label map: {label_map}")

    # 5. Attach results to df
    df["predicted_regime"]    = final_regimes
    df["regime_label"]        = final_labels
    df["mmd"]                 = mmd_series
    df["transition_flag"]     = trans_flag

    # Confidence: ratio of MMD to threshold (capped at 1)
    mmd_thresh = threshold  # rename for clarity
    conf_raw = mmd_series / (mmd_thresh * 2)
    df["confidence"] = pd.Series(np.clip(conf_raw, 0, 1)).fillna(0.5).values

    # Regime statistics
    regime_ret_map, regime_vol_map = {}, {}
    for rid in sorted(df["predicted_regime"].unique()):
        subset = df[df["predicted_regime"] == rid]
        regime_ret_map[rid] = float(subset["log_ret"].mean())
        regime_vol_map[rid] = float(subset["ret_std_20d"].mean())

    df["regime_return_forecast"] = df["predicted_regime"].map(regime_ret_map)
    df["regime_vol_forecast"]    = df["predicted_regime"].map(regime_vol_map)

    # 6. Filter to evaluation window
    eval_start = pd.to_datetime(EVAL_START)
    eval_end   = pd.to_datetime(EVAL_END)
    df_eval = df[(df["date"] >= eval_start) & (df["date"] <= eval_end)].copy()
    print(f"\n[6] Evaluation window: {EVAL_START} → {EVAL_END}  ({len(df_eval)} rows)")

    # 7. Write CSV
    out_csv = os.path.join(OUT_DIR, f"{ALGO_NAME}_regimes.csv")
    out_cols = ["date", "predicted_regime", "confidence", "transition_flag",
                "regime_label", "regime_return_forecast", "regime_vol_forecast"]
    df_eval[out_cols].to_csv(out_csv, index=False, date_format="%Y-%m-%d")
    print(f"\n[7] Wrote: {out_csv}  ({len(df_eval)} rows)")

    # 8. Write metadata JSON
    num_regimes = int(df["predicted_regime"].max()) + 1
    sorted_rids = sorted(df["predicted_regime"].unique())
    regime_labels_map = {str(int(rid)): str(final_labels[np.where(final_regimes == int(rid))[0][0]])
                          for rid in sorted_rids
                          if np.where(final_regimes == int(rid))[0].size > 0}

    meta = {
        "algorithm": ALGO_NAME,
        "algorithm_full_name": "Signature-based MMD Regime Detector with K-Means",
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
    print(f"[8] Wrote: {out_meta}")

    # 9. Summary
    print("\n[9] Regime summary:")
    for rid in sorted(df["predicted_regime"].unique()):
        label  = label_map.get(rid, "?")
        n_days = (df["predicted_regime"] == rid).sum()
        print(f"    Regime {rid} ({label}): {n_days} days, "
              f"avg_ret={regime_ret_map.get(rid, 0):.6f}, "
              f"avg_vol={regime_vol_map.get(rid, 0):.6f}")

    n_trans = int(df["transition_flag"].sum())
    print(f"\n    Total regime transitions: {n_trans}")
    print(f"    MMD threshold used: {threshold:.6f}")

    # 10. Sample rows
    print("\n[10] Sample output (first 5 rows):")
    print(df_eval[["date","predicted_regime","confidence","transition_flag","regime_label"]].head().to_string(index=False))

    # Last 5 rows
    print("\n    Last 5 rows:")
    print(df_eval[["date","predicted_regime","confidence","transition_flag","regime_label"]].tail().to_string(index=False))

    print("\n[DONE]")
    return df_eval, meta

if __name__ == "__main__":
    main()
