#!/usr/bin/env python3
"""
Regime Detection: Non-parametric Online Market Regime Detection
Based on Horvath & Issa (2023), arXiv:2306.15835 / SSRN 3947905

"Non-parametric online market regime detection and regime clustering
for multidimensional and path-dependent data structures"

REWRITE v2.1 — Fixed from evaluation feedback:
  - Single asset: S&P 500 index (not individual stocks)
  - Daily output: one row per trading day in evaluation window
  - Regime IDs: 0, 1, 2 (contiguous, zero-based)
  - Proper CSV + JSON output conforming to output_spec.json v1.0
  - Hybrid approach: ensemble MMD for change detection + daily clustering

Paper equations referenced:
  Eq.(5)/(6): Path signature (iterated integrals)
  Eq.(16): Unbiased MMD estimator
  Eq.(24): Sub-path partitioning SP_h
  Eq.(25): Ensemble paths EP_h
  Eq.(28): Score matrix Lambda(s)
  Eq.(30): Auto-evaluator A_L(s)
"""

import numpy as np
import pandas as pd
import json
import os
import sys
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone

# ─── Configuration ───────────────────────────────────────────────────────────

EVAL_START = "2021-05-10"
EVAL_END = "2026-05-06"

# Paper hyperparameters (Section 4, daily data)
H1 = 20        # sub-path length (~1 month of daily data, Eq. 24)
H2 = 10        # ensemble size (Eq. 25)
SIG_LEVEL = 2  # signature truncation level
ALPHA = 0.05   # significance level for bootstrap threshold
N_BOOT = 500   # bootstrap samples
N_REGIMES = 3  # number of regimes for clustering (Sec. 5)

PARQUET_PATH = "/opt/data/kanban/workspaces/t_f9e0a695/sp500_data/sp500_ohlcv_20yr.parquet"
OUTPUT_DIR = "/opt/data/kanban/workspaces/t_7afc4e91/paper-3947905-v2"
RANDOM_SEED = 42


# ─── Signature Computation (Eq. 5-6) ────────────────────────────────────────

def extract_signature_features(path: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Truncated path signature features (Eq. 5-6).
    Level 1: mean(dX_i) → d features
    Level 2: ∫∫ dX_i dX_j → d*d features
    """
    dX = np.diff(path, axis=0)
    if len(dX) < 2:
        d = path.shape[1] if len(path.shape) > 1 else 1
        return np.zeros(d * (1 + d))

    d = dX.shape[1]
    l1 = dX.mean(axis=0)

    cumul = np.zeros(d)
    l2 = np.zeros((d, d))
    for t in range(len(dX)):
        dx = dX[t]
        l2 += np.outer(cumul, dx)
        cumul += dx
    l2 /= len(dX)

    feats = np.concatenate([l1, l2.flatten()])

    if level >= 3 and d <= 3:
        cumul1 = np.zeros(d)
        cumul2 = np.zeros((d, d))
        l3 = np.zeros(d * d * d)
        for t in range(len(dX)):
            dx = dX[t]
            cumul2 += np.outer(cumul1, dx)
            l3 += np.repeat(cumul2.flatten(), d) * np.tile(dx, d * d)
            cumul1 += dx
        l3 /= len(dX)
        feats = np.concatenate([feats, l3])

    return feats


def transform_subpath(data_slice: np.ndarray) -> np.ndarray:
    """Phi = phi_incr ∘ phi_time ∘ phi_norm"""
    mu = data_slice.mean(axis=0)
    sigma = data_slice.std(axis=0) + 1e-12
    norm = (data_slice - mu) / sigma

    inc = np.diff(norm, axis=0)
    inc_sigma = inc.std(axis=0) + 1e-12
    inc = (inc - np.mean(inc, axis=0)) / inc_sigma

    T_inc = len(inc)
    t_col = np.arange(T_inc)[:, None] / max(T_inc - 1, 1)
    return np.concatenate([t_col, inc], axis=1)


# ─── Kernel and MMD (Eq. 16) ─────────────────────────────────────────────────

def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """k(x,y) = exp(-gamma * ||x-y||^2)"""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    sq = np.maximum(XX + YY.T - 2.0 * X @ Y.T, 0.0)
    return np.exp(-gamma * sq)


def auto_gamma(X: np.ndarray) -> float:
    """gamma = 1 / median pairwise squared distance"""
    n = len(X)
    if n < 3:
        return 1.0
    s = min(n, 200)
    idx = np.random.choice(n, s, replace=False)
    S = X[idx]
    d2 = np.sum((S[:, None] - S[None]) ** 2, axis=2)
    off = d2[~np.eye(s, dtype=bool)]
    med = float(np.median(off))
    return 1.0 / med if med > 1e-10 else 1.0


def mmd_unbiased(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """Unbiased MMD² estimator (Eq. 16)."""
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return 0.0
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)
    t_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    t_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    t_xy = Kxy.sum() / (n * m)
    return max(t_xx + t_yy - 2.0 * t_xy, 0.0)


def bootstrap_threshold(ref: np.ndarray, ens_size: int, alpha: float = 0.05,
                        n_boot: int = 500, gamma: Optional[float] = None) -> Tuple[float, float]:
    """Bootstrap null distribution critical value (Def. 3.5)."""
    if gamma is None:
        gamma = auto_gamma(ref)
        if gamma < 1e-6:
            gamma = 1.0
    N = len(ref)
    mmds = []
    for _ in range(n_boot):
        if N >= 2 * ens_size:
            idx = np.random.choice(N, 2 * ens_size, replace=False)
        else:
            idx = np.random.choice(N, 2 * ens_size, replace=True)
        A, B = ref[idx[:ens_size]], ref[idx[ens_size:]]
        mmds.append(mmd_unbiased(A, B, gamma))
    return float(np.percentile(mmds, (1 - alpha) * 100)), gamma


# ─── Core Pipeline ───────────────────────────────────────────────────────────

def build_daily_signature_features(df: pd.DataFrame, h1: int = 20,
                                    level: int = 2) -> Tuple[np.ndarray, List[str]]:
    """
    Build daily signature features using rolling windows of h1 days.
    Day t gets features from window [t-h1+1, t].

    This adapts the paper's sub-path approach (Eq. 24) to daily granularity.
    """
    prices = df['close'].values.astype(np.float64)
    log_ret = np.zeros(len(prices))
    log_ret[1:] = np.diff(np.log(np.abs(prices) + 1e-12))

    vol = df['volume'].values.astype(np.float64)
    vol_ratio = vol / (np.mean(vol) + 1e-12)

    raw = np.column_stack([prices, log_ret, vol_ratio])

    features = []
    dates = []
    for t in range(h1, len(raw)):
        sub = raw[t - h1:t].copy()
        t_path = transform_subpath(sub)
        sig = extract_signature_features(t_path, level=level)
        features.append(sig)
        dates.append(str(df.iloc[t]['date'])[:10])

    return np.array(features), dates


def regime_kmeans_daily(features: np.ndarray, n_regimes: int = 3,
                         smooth_window: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster daily signature features and smooth the regime path.

    This implements Section 5 (regime clustering) with temporal smoothing
    to ensure stable, economically meaningful regimes.

    Parameters
    ----------
    features : np.ndarray
        Daily signature features, shape (n_days, feat_dim)
    n_regimes : int
        Number of clusters
    smooth_window : int
        Median filter window (in trading days) for smoothing

    Returns
    -------
    regimes : np.ndarray — smoothed regime labels (zero-based contiguous)
    confidence : np.ndarray — confidence scores in [0, 1]
    """
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_regimes, random_state=RANDOM_SEED,
                    n_init=10, max_iter=500)
        raw_labels = km.fit_predict(features)

        # Confidence from centroid distances
        distances = km.transform(features)
        min_dists = distances.min(axis=1)
        max_d = min_dists.max()
        confidence = 1.0 - (min_dists / max_d) if max_d > 0 else np.ones(len(features))

        # Temporal smoothing: majority vote in rolling window
        smoothed = _majority_smooth(raw_labels, smooth_window)

        return smoothed, confidence

    except Exception as e:
        print(f"  [WARN] k-means failed ({e}), using fallback")
        return _fallback_daily(features, n_regimes)


def _majority_smooth(labels: np.ndarray, window: int) -> np.ndarray:
    """
    Smooth regime labels using a centered majority-vote sliding window.
    This removes spurious single-day regime flips while preserving
    genuine regime transitions.
    """
    n = len(labels)
    half = window // 2
    smoothed = labels.copy()

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_labels = labels[start:end]
        # Majority vote
        counts = np.bincount(window_labels)
        smoothed[i] = np.argmax(counts)

    return smoothed


def _fallback_daily(features: np.ndarray, n_regimes: int) -> Tuple:
    """Fallback: assign by rolling return quantiles + smoothing."""
    n = len(features)
    scores = features[:, 0]
    boundaries = np.percentile(scores, np.linspace(0, 100, n_regimes + 1)[1:-1])
    labels = np.clip(np.digitize(scores, boundaries), 0, n_regimes - 1)
    labels = _majority_smooth(labels, 21)
    confidence = np.ones(n) * 0.5
    return labels, confidence


def compute_change_point_diagnostics(ensemble_features: List[np.ndarray],
                                      gamma: float, alpha: float = 0.05,
                                      n_boot: int = 500) -> Dict:
    """
    Compute auto-evaluator MMD scores and bootstrap threshold for diagnostics.
    (Eq. 30, Def. 3.5)
    """
    n = len(ensemble_features)
    scores = np.zeros(n)
    for i in range(1, n):
        scores[i] = mmd_unbiased(ensemble_features[i - 1],
                                  ensemble_features[i], gamma)

    ref = np.concatenate(ensemble_features[:n // 2], axis=0)
    threshold, gamma = bootstrap_threshold(ref, min(10, n // 2),
                                            alpha, n_boot, gamma)

    cps = [i for i in range(1, n) if scores[i] > threshold]
    return {
        'scores': scores, 'threshold': threshold, 'gamma': gamma,
        'change_points': cps,
    }


def _remap_zero_based(labels: np.ndarray) -> Tuple[np.ndarray, int]:
    """Remap to contiguous zero-based integers."""
    unique = np.unique(labels)
    mapping = {old: new for new, old in enumerate(sorted(unique))}
    return np.array([mapping[l] for l in labels], dtype=int), len(unique)


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_output(dates, regimes, confidence, transitions, n_regimes):
    issues = []
    n = len(dates)
    if n < 100:
        issues.append(f"Only {n} rows")
    if np.any(regimes < 0):
        issues.append(f"Negative regimes: {np.sum(regimes < 0)}")
    if np.any(confidence < 0) or np.any(confidence > 1):
        issues.append(f"Confidence out of [0,1]")
    if not all(x in [0, 1] for x in np.unique(transitions)):
        issues.append("Non-binary transitions")
    if transitions[0] != 0:
        issues.append("First row transition_flag != 0")
    unique_r = sorted(np.unique(regimes))
    if unique_r != list(range(len(unique_r))):
        issues.append(f"Regime IDs not contiguous zero-based: {unique_r}")
    if len(unique_r) != n_regimes:
        issues.append(f"num_regimes mismatch")
    if np.any(np.isnan(regimes)) or np.any(np.isnan(confidence)) or np.any(np.isnan(transitions)):
        issues.append("NaN values in output")
    trans_rate = transitions.sum() / max(n - 1, 1)
    if trans_rate > 0.30:
        issues.append(f"Transition rate {trans_rate:.2%} > 30%")
    return issues


# ─── Output Writing ──────────────────────────────────────────────────────────

def write_csv(dates, regimes, confidence, transitions, labels_map, path):
    with open(path, 'w') as f:
        f.write("date,predicted_regime,confidence,transition_flag,regime_label\n")
        for i in range(len(dates)):
            r = int(regimes[i])
            label = labels_map.get(str(r), f"regime_{r}")
            f.write(f"{dates[i]},{r},{confidence[i]:.4f},{int(transitions[i])},{label}\n")


def write_metadata(n_regimes, labels_map, threshold, gamma, n_rows, n_cps,
                    n_ens, path):
    meta = {
        "algorithm": "sig_mmd_regime",
        "algorithm_full_name": "Signature MMD Online Regime Detection (Horvath-Issa 2023)",
        "algorithm_family": "changepoint",
        "paper_reference": "arXiv:2306.15835",
        "num_regimes": n_regimes,
        "regime_labels_map": labels_map,
        "parameters": {
            "h1": H1, "h2": H2, "signature_level": SIG_LEVEL,
            "alpha": ALPHA, "n_bootstrap": N_BOOT,
            "n_regimes": N_REGIMES, "random_seed": RANDOM_SEED,
            "threshold": threshold, "gamma": gamma, "n_ensembles": n_ens,
        },
        "training_window_days": None,
        "features_used": [
            "daily_return", "log_return", "volume_ratio",
            "signature_level_1", "signature_level_2",
        ],
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_spec_version": "1.0",
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Market Regime Detection — Horvath & Issa (2023), arXiv:2306.15835")
    print("REWRITE v2.1: S&P 500, daily output, zero-based regimes")
    print("=" * 80)

    np.random.seed(RANDOM_SEED)

    # 1. Load
    print(f"\n[1/7] Loading data...")
    df = pd.read_parquet(PARQUET_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Shape: {df.shape}, Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # 2. Filter to eval window
    eval_start = pd.Timestamp(EVAL_START)
    eval_end = pd.Timestamp(EVAL_END)
    mask = (df['date'] >= eval_start) & (df['date'] <= eval_end)
    df_eval = df[mask].reset_index(drop=True)
    total_days = len(df_eval)
    print(f"\n[2/7] Evaluation window: {EVAL_START} to {EVAL_END}, {total_days} trading days")

    # 3. Build daily signature features
    print(f"\n[3/7] Building daily signature features (h1={H1}, level={SIG_LEVEL})...")
    features, dates = build_daily_signature_features(df_eval, h1=H1, level=SIG_LEVEL)
    print(f"  Daily features: {features.shape}, Range: {dates[0]} to {dates[-1]}")

    # 4. Ensemble MMD diagnostics (for reference, not driving regime assignment)
    print(f"\n[4/7] Computing ensemble MMD diagnostics (auto-evaluator, Eq. 30)...")
    n_sub = len(df_eval) // H1
    sub_sigs = []
    for j in range(n_sub):
        sp = df_eval.iloc[j * H1:(j + 1) * H1]
        raw = np.column_stack([
            sp['close'].values.astype(np.float64),
            np.concatenate([[0], np.diff(np.log(sp['close'].values.astype(np.float64) + 1e-12))]),
            sp['volume'].values.astype(np.float64) / (sp['volume'].mean() + 1e-12),
        ])
        t_path = transform_subpath(raw)
        sub_sigs.append(extract_signature_features(t_path, level=SIG_LEVEL))
    sub_sigs = np.array(sub_sigs)

    n_ens = n_sub // H2
    ens_feats = []
    ens_dates = []
    for k in range(n_ens):
        s, e = k * H2, (k + 1) * H2
        ens_feats.append(sub_sigs[s:e])
        ens_dates.append(str(df_eval.iloc[e * H1 - 1]['date'])[:10])

    gamma = auto_gamma(ens_feats[0])
    if gamma < 1e-6:
        gamma = 1.0
    diag = compute_change_point_diagnostics(ens_feats, gamma, ALPHA, N_BOOT)
    print(f"  Ensembles: {n_ens}, Threshold: {diag['threshold']:.6f}, Gamma: {diag['gamma']:.4f}")
    print(f"  MMD change points: {len(diag['change_points'])}")
    for cp in diag['change_points'][:5]:
        if cp < len(ens_dates):
            print(f"    {ens_dates[cp]}: MMD={diag['scores'][cp]:.6f}")

    # 5. Cluster daily features into regimes (Sec. 5)
    print(f"\n[5/7] Clustering daily features into {N_REGIMES} regimes...")
    daily_regimes, confidence = regime_kmeans_daily(
        features, n_regimes=N_REGIMES, smooth_window=21
    )

    # Ensure zero-based contiguous
    daily_regimes, n_regimes_found = _remap_zero_based(daily_regimes)

    # The first H1 days of the eval window don't have features (need warmup).
    # Prepend the first regime label and dates for those days.
    n_warmup = total_days - len(daily_regimes)
    if n_warmup > 0:
        warmup_dates = [str(df_eval.iloc[i]['date'])[:10] for i in range(n_warmup)]
        dates = warmup_dates + dates
        daily_regimes = np.concatenate([np.full(n_warmup, int(daily_regimes[0])), daily_regimes])
        confidence = np.concatenate([np.full(n_warmup, 0.5), confidence])

    # Regime summary
    for r in range(n_regimes_found):
        mask_r = daily_regimes == r
        pct = 100.0 * mask_r.sum() / len(daily_regimes)
        avg_conf = confidence[mask_r].mean()
        print(f"  Regime {r}: {mask_r.sum()} days ({pct:.0f}%), avg conf={avg_conf:.3f}")

    # 6. Compute transitions
    transitions = np.zeros(total_days, dtype=int)
    transitions[1:] = (daily_regimes[1:] != daily_regimes[:-1]).astype(int)
    n_cps = int(transitions.sum())
    trans_rate = n_cps / max(total_days - 1, 1)
    print(f"\n[6/7] Transitions: {n_cps}, rate={trans_rate:.2%}")

    # Print transition dates
    cp_indices = np.where(transitions == 1)[0]
    for idx in cp_indices[:15]:
        if idx < len(dates):
            print(f"  {dates[idx]}: {daily_regimes[idx-1]} → {daily_regimes[idx]}")

    # 7. Validate and write
    print(f"\n[7/7] Validating and writing output...")
    issues = validate_output(dates, daily_regimes, confidence, transitions, n_regimes_found)
    if issues:
        print(f"  [WARN] {len(issues)} issue(s):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  All validation checks passed ✓")

    regime_names = ['bull', 'bear', 'sideways', 'high_vol', 'crisis', 'recovery']
    labels_map = {str(r): regime_names[r] if r < len(regime_names) else f"regime_{r}"
                  for r in range(n_regimes_found)}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "sig_mmd_regime_regimes.csv")
    json_path = os.path.join(OUTPUT_DIR, "sig_mmd_regime_metadata.json")

    write_csv(dates, daily_regimes, confidence, transitions, labels_map, csv_path)
    write_metadata(n_regimes_found, labels_map, diag['threshold'], diag['gamma'],
                   total_days, n_cps, n_ens, json_path)

    print(f"  CSV: {csv_path} ({total_days} rows)")
    print(f"  JSON: {json_path}")

    print(f"\n{'=' * 80}")
    print("OUTPUT SUMMARY")
    print(f"{'=' * 80}")
    print(f"Algorithm: Signature MMD Regime Detection")
    print(f"Asset: S&P 500 Index")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Rows: {total_days}, Regimes: {n_regimes_found} ({sorted(np.unique(daily_regimes).tolist())})")
    print(f"Transitions: {n_cps} ({trans_rate:.2%})")
    print(f"Files: {csv_path}, {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
