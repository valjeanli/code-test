#!/usr/bin/env python3
"""
code_coder-kimi.py

Implementation of non-parametric online market regime detection using
truncated path signatures and Maximum Mean Discrepancy (MMD) two-sample testing.

Based on the paper:
    "Non-parametric online market regime detection and regime clustering
     for multidimensional and path-dependent data structures"
    Zacharia Issa, Blanka Horvath
    arXiv:2306.15835 (SSRN 3947905), 2023.

Methodology:
    1. Load S&P 500 daily close prices.
    2. Pre-process: log-increment transform.
    3. Build rolling path ensembles (sliding windows of length L, stride=5).
    4. Compute truncated path signatures for each path via iisignature.
    5. Compute unbiased MMD^2 between reference and test ensembles.
    6. Detect regime change points where MMD exceeds a global percentile threshold
       with non-maximum suppression.
    7. Output detected transitions with dates and MMD trajectory.

Dependencies: numpy, scipy, sklearn, matplotlib, iisignature
"""
from __future__ import annotations

import csv
import os
import sys
from typing import List, Tuple

import numpy as np

# Optional plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# 1.  DATA LOADING
# ---------------------------------------------------------------------------

def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Date,Close CSV; return dates (str), closes (float)."""
    dates, closes = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row["Date"])
            closes.append(float(row["Close"]))
    return np.array(dates), np.array(closes, dtype=float)


# ---------------------------------------------------------------------------
# 2.  PRE-PROCESSING  (cf. paper Section 4)
# ---------------------------------------------------------------------------
# For a price path P(t), we first take log-prices X(t) = log P(t).
# The discrete increments are delta X_i = log(P_i) - log(P_{i-1}).
# When feeding to the signature library we supply the sequence of increments
# as a path of shape (L, d).  Time normalisation is implicit because we treat
# each path as L equally spaced steps.
# ---------------------------------------------------------------------------

def build_paths(
    prices: np.ndarray,
    L: int = 21,
    stride: int = 5,
    dates: np.ndarray | None = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Compute log-returns, then slide a window of length L to create streamed paths.

    Returns
    -------
    paths : list of np.ndarray, each shape (L, 1)
    path_dates : np.ndarray of end-dates for each path window
    """
    loginc = np.diff(np.log(prices))
    paths = []
    path_dates = []
    for i in range(0, len(loginc) - L + 1, stride):
        seg = loginc[i : i + L]
        paths.append(seg.reshape(-1, 1))
        if dates is not None:
            path_dates.append(dates[i + L])  # end date of the L-day window
    return paths, np.array(path_dates)


# ---------------------------------------------------------------------------
# 3.  SIGNATURE FEATURES  (cf. paper Section 2 & 5)
# ---------------------------------------------------------------------------
# We use a *truncated* signature map S_M( X ) where M is the truncation depth.
# Although the paper strongly advocates the full signature kernel (computed via
# a Goursat PDE), the truncated map is the pragmatic implementation pathway
# and was benchmarked in the paper as MMD-T.
# ---------------------------------------------------------------------------

def signature_features(paths: List[np.ndarray], depth: int = 4) -> np.ndarray:
    """
    Compute the truncated signature (depth=depth) for each path in *paths*.

    Parameters
    ----------
    paths : list of np.ndarray, each shape (L, d)
    depth : int
        Truncation level of the signature.

    Returns
    -------
    feats : np.ndarray, shape (n_paths, sig_dim)
    """
    import iisignature as ii
    sigs = [ii.sig(p, depth) for p in paths]
    return np.vstack(sigs)


# ---------------------------------------------------------------------------
# 4.  MMD TWO-SAMPLE TEST  (cf. paper Section 2.3, eq. (11)-(14))
# ---------------------------------------------------------------------------
# Unbiased U-statistic estimator of MMD^2 in a feature space H.
# For a linear kernel on truncated signatures this is the Euclidean inner
# product between signature vectors.
# ---------------------------------------------------------------------------

def mmd2_unbiased(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Compute unbiased MMD^2 estimate between two ensembles of feature vectors.

    Parameters
    ----------
    xs : np.ndarray, shape (m, d)
    ys : np.ndarray, shape (n, d)

    Returns
    -------
    mmd2 : float
    """
    m, n = xs.shape[0], ys.shape[0]
    if m < 2 or n < 2:
        return 0.0
    kxx = xs @ xs.T
    kyy = ys @ ys.T
    kxy = xs @ ys.T
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(
        kxx.sum() / (m * (m - 1))
        + kyy.sum() / (n * (n - 1))
        - 2.0 * kxy.sum() / (m * n)
    )


# ---------------------------------------------------------------------------
# 5.  ONLINE REGIME DETECTOR  (cf. paper Section 4, Algorithm MMD-DET)
# ---------------------------------------------------------------------------
# At each step we hold a *reference* ensemble and a *test* ensemble.
# We compute MMD(reference, test).  If MMD exceeds a threshold we declare a
# regime change.  The original paper advocates a bootstrap threshold per step;
# for computational stability on long time series we use a global percentile
# of the MMD score series combined with non-maximum suppression.
# ---------------------------------------------------------------------------

def detect_regime_changes(
    feats: np.ndarray,
    path_dates: np.ndarray,
    n_reference: int = 10,
    n_test: int = 10,
    step: int = 2,
    threshold_percentile: float = 98.0,
    suppression_width: int = 20,
) -> Tuple[List[Tuple[str, float, float]], np.ndarray, np.ndarray]:
    """
    Run sliding-window regime detection.

    Parameters
    ----------
    feats : np.ndarray, shape (N, d)
        Signature feature vectors for each path.
    path_dates : np.ndarray of str, shape (N,)
        End dates aligned to each path window.
    n_reference : int
        Number of paths in the reference ensemble.
    n_test : int
        Number of paths in the test ensemble.
    step : int
        Sliding step (in path indices).
    threshold_percentile : float
        Global percentile used as critical threshold.
    suppression_width : int
        After a detection, skip this many scores to avoid duplicates.

    Returns
    -------
    change_points : list of (date, mmd2, threshold)
    scores : np.ndarray
    score_dates : np.ndarray
    """
    scores = []
    score_dates = []
    i = 0
    while i + n_reference + n_test <= feats.shape[0]:
        ref = feats[i : i + n_reference]
        test = feats[i + n_reference : i + n_reference + n_test]
        scores.append(mmd2_unbiased(ref, test))
        # End date of the test window
        end_idx = i + n_reference + n_test - 1
        score_dates.append(str(path_dates[end_idx]))
        i += step

    scores = np.array(scores)
    threshold = float(np.percentile(scores, threshold_percentile))

    change_points = []
    skip_until = -1
    for j in range(len(scores)):
        if j < skip_until:
            continue
        if scores[j] > threshold:
            change_points.append((score_dates[j], float(scores[j]), threshold))
            skip_until = j + suppression_width

    return change_points, scores, np.array(score_dates)


# ---------------------------------------------------------------------------
# 6.  AGGLOMERATIVE REGIME CLUSTERING  (cf. paper Section 5)
# ---------------------------------------------------------------------------
# For the regime *clustering* problem (MRCP) the paper uses hierarchical
# agglomerative clustering with the signature-MMD as a pairwise distance.
# ---------------------------------------------------------------------------

def regime_clustering(
    paths: List[np.ndarray],
    n_clusters: int = 3,
    sig_depth: int = 4,
    linkage: str = "average",
) -> np.ndarray:
    """
    Hierarchical clustering of path ensembles using MMD distance matrix.

    Returns
    -------
    labels : np.ndarray, shape (len(paths),)
    """
    from sklearn.cluster import AgglomerativeClustering

    feats = signature_features(paths, depth=sig_depth)
    mu = feats.mean(axis=0, keepdims=True)
    sigma = feats.std(axis=0, keepdims=True) + 1e-12
    feats = (feats - mu) / sigma

    n = feats.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = max(0.0, mmd2_unbiased(feats[i : i + 1], feats[j : j + 1]))
            dist[i, j] = dist[j, i] = np.sqrt(d)

    model = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage=linkage
    )
    return model.fit_predict(dist)


# ---------------------------------------------------------------------------
# 7.  VISUALISATION  (cf. paper Figure 6, Figure 11)
# ---------------------------------------------------------------------------

def plot_results(
    closes: np.ndarray,
    all_dates: np.ndarray,
    score_dates: np.ndarray,
    scores: np.ndarray,
    change_points: List[Tuple[str, float, float]],
    out_path: str = "regime_detection.png",
) -> None:
    if not HAS_MPL:
        print("[WARN] matplotlib unavailable; skipping plot.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Price series (thin black line)
    n = len(closes)
    step = max(1, n // 5000)
    ax1.plot(
        range(0, n, step), closes[::step], color="black", alpha=0.3, label="S&P 500 Close"
    )
    ax1.set_ylabel("Close Price (USD)", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    # Map score dates to indices in all_dates for alignment
    score_indices = []
    for d in score_dates:
        idxs = np.where(all_dates == d)[0]
        score_indices.append(int(idxs[0]) if len(idxs) else n - 1)

    ax2 = ax1.twinx()
    ax2.plot(
        score_indices, scores, color="tab:blue", lw=1.2, label="MMD² score"
    )

    # Threshold line
    if change_points:
        thresh = change_points[0][2]
        ax2.axhline(
            thresh, color="tab:red", lw=1.0, linestyle="--",
            label=f"Threshold (p={98:g}th pct)",
        )

    # Mark detected change points
    for cp_date, cp_score, _ in change_points:
        idxs = np.where(all_dates == cp_date)[0]
        cp_idx = int(idxs[0]) if len(idxs) else None
        if cp_idx is not None:
            ax2.axvline(cp_idx, color="tab:green", lw=1.2, alpha=0.5)
            ax2.plot(cp_idx, cp_score, "o", color="tab:green", markersize=5)

    ax2.set_ylabel("MMD²", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_xlabel("Time index (days)")
    plt.title("Signature-MMD Online Regime Detection — S&P 500")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[INFO] Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# 8.  MAIN PIPELINE
# ---------------------------------------------------------------------------

def main() -> int:
    # Resolve data file (one level above the script directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.normpath(os.path.join(script_dir, "..", "sp500.csv"))
    out_dir = script_dir

    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        return 1

    dates, closes = load_csv(data_path)
    print(
        f"[INFO] Loaded {len(closes)} daily S&P 500 observations "
        f"({dates[0]} to {dates[-1]})."
    )

    # Build streamed paths
    L = 21  # ~1 trading month
    stride = 5
    paths, path_dates = build_paths(closes, L=L, stride=stride, dates=dates)
    print(f"[INFO] Built {len(paths)} streamed paths (length={L}, stride={stride}).")

    # Compute truncated signature features
    sig_depth = 4
    print(f"[INFO] Computing truncated signatures (depth={sig_depth}) ...")
    feats = signature_features(paths, depth=sig_depth)
    print(f"[INFO] Signature dimension = {feats.shape[1]}")

    # Normalise features for numerical stability
    mu = feats.mean(axis=0, keepdims=True)
    sigma = feats.std(axis=0, keepdims=True) + 1e-12
    feats = (feats - mu) / sigma

    # Run detector
    print("[INFO] Running sliding-window MMD detector ...")
    change_points, scores, score_dates = detect_regime_changes(
        feats,
        path_dates,
        n_reference=10,
        n_test=10,
        step=2,
        threshold_percentile=98.0,
        suppression_width=20,
    )
    print(f"[INFO] {len(change_points)} regime change point(s) detected.")
    for cp_date, cp_score, cp_thresh in change_points:
        print(
            f"  Date: {cp_date}  |  MMD²={cp_score:.4f}  >  threshold={cp_thresh:.4f}"
        )

    # Clustering (for completeness)
    print("\n[INFO] Running regime clustering (k=3) ...")
    cluster_labels = regime_clustering(paths, n_clusters=3, sig_depth=sig_depth)
    uniq, counts = np.unique(cluster_labels, return_counts=True)
    print(f"[INFO] Cluster counts: {dict(zip(uniq, counts))}")

    # Save textual output
    out_txt = os.path.join(out_dir, "output_coder-kimi.txt")
    with open(out_txt, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Signature-MMD Regime Detection Output\n")
        f.write("Paper: arXiv:2306.15835 (SSRN 3947905)\n")
        f.write("=" * 70 + "\n\n")

        f.write("--- CONFIGURATION ---\n")
        f.write(f"  path_length: {L}\n")
        f.write(f"  stride: {stride}\n")
        f.write(f"  n_reference: 10\n")
        f.write(f"  n_test: 10\n")
        f.write(f"  sig_depth: {sig_depth}\n")
        f.write(f"  threshold_type: global_98th_percentile\n")
        f.write(f"  suppression_width: 20\n")
        f.write(f"  data_points: {len(closes)}\n")
        f.write(f"  paths_generated: {len(paths)}\n")
        f.write(f"  mmd_scores_computed: {len(scores)}\n\n")

        f.write("--- DETECTED REGIME CHANGE POINTS ---\n")
        if change_points:
            for cp_date, cp_score, cp_thresh in change_points:
                f.write(
                    f"  Date: {cp_date}  |  MMD²: {cp_score:.6f}  |  "
                    f"threshold: {cp_thresh:.6f}\n"
                )
        else:
            f.write("  No change points detected at the chosen threshold.\n")

        f.write("\n--- MMD SCORE TRAJECTORY (first 50) ---\n")
        for i in range(min(50, len(scores))):
            f.write(f"  {score_dates[i]}  MMD²={scores[i]:.6f}\n")
        if len(scores) > 50:
            f.write(f"  ... ({len(scores) - 50} more rows)\n")

        f.write("\n--- MMD SCORE TRAJECTORY (last 50) ---\n")
        for i in range(max(0, len(scores) - 50), len(scores)):
            f.write(f"  {score_dates[i]}  MMD²={scores[i]:.6f}\n")

        f.write("\n--- REGIME CLUSTERING SUMMARY ---\n")
        for k, v in zip(uniq, counts):
            f.write(f"  Cluster {k}: {v} paths\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"\n[INFO] Output saved to {out_txt}")

    # Plot
    plot_path = os.path.join(out_dir, "regime_detection.png")
    plot_results(closes, dates, score_dates, scores, change_points, out_path=plot_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
