#!/usr/bin/env python3
"""
ACC (Asset Clustering through Correlation) — Paper 2103.14506
Tang, Xu, Zhou — Columbia University

Implements:
  1. CORD dissimilarity matrix (fully vectorized ~1s for d=500)
  2. PARTITION procedure via union-find (single-linkage thresholding)
  3. ACC algorithm with ε-tuning (Rules 1-3 from paper)
  4. Asset selection: lowest variance per cluster (Theorem 2)

Usage: python code_coder-minimax.py
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import time

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────── Parameters ────────────────────────────────────
PARAM_WINDOW   = 504    # lookback window (trading days; must exceed d for full rank)
PARAM_NG       = 40     # ε grid resolution
PARAM_K_MIN    = 10     # Rule 3: minimum clusters
PARAM_K_MAX    = 35     # Rule 3: maximum clusters
PARAM_EPS_MIN  = 0.02   # ε search lower bound
PARAM_EPS_MAX  = 1.50   # ε search upper bound

DATA_DIR    = "/opt/data/kanban/workspaces/t_dc995cda/sp500_data"
OUT_FILE    = Path("/opt/data/kanban/workspaces/t_8fcc6dd0/paper-2103.14506/output_coder-minimax.txt")

# ─────────────────────────── Data Loading ────────────────────────────────────
def load_returns(data_dir=DATA_DIR):
    df = pd.read_parquet(Path(data_dir) / "sp500_ohlcv_5yr.parquet")
    wide = df.pivot_table(index="Date", columns="symbol", values="Close")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    wide.columns.name = None
    rets = wide.pct_change().dropna(how="all")
    print(f"Returns: {rets.shape}, tickers={rets.columns.nunique()}, "
          f"dates: {rets.index.min().date()} → {rets.index.max().date()}")
    return rets

# ─────────────────────────── CORD Matrix ─────────────────────────────────────
def compute_cord(rho_hat):
    """
    CORD(i,j) = max_{l != i, j} |rho_il - rho_jl|

    Vectorized O(d^3) memory, ~1s for d=500.
    diffs[i,j,l] = |rho[i,l] - rho[j,l|], then max over l,
    masking l=i and l=j.
    """
    d = rho_hat.shape[0]
    diffs = np.abs(rho_hat[:, None, :] - rho_hat[None, :, :])
    idx = np.arange(d)
    diffs[idx, :, idx] = -np.inf   # l == i
    diffs[idx, idx, :] = -np.inf   # l == j
    cord = np.max(diffs, axis=2)
    np.fill_diagonal(cord, 0.0)
    return cord

# ─────────────────────────── PARTITION Procedure ────────────────────────────
def partition_procedure(cord, eps):
    """
    PARTITION procedure (Procedure 1, paper).
    Uses union-find (disjoint set union) to compute connected components
    of the CORD graph thresholded at eps: edge (i,j) exists iff cord[i,j] <= eps.
    This implements single-linkage clustering with threshold eps.

    Complexity: O(d^2) to build adjacency + near-O(d^2) for unions.
    """
    d = cord.shape[0]
    parent = np.arange(d)
    rank = np.zeros(d, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1

    i_idx, j_idx = np.where(cord <= eps)
    for ii, jj in zip(i_idx, j_idx):
        union(ii, jj)

    comps = {}
    for i in range(d):
        r = find(i)
        if r not in comps:
            comps[r] = []
        comps[r].append(i)

    clusters = [sorted(c) for c in comps.values()]
    clusters.sort(key=lambda x: (-len(x), x[0]))
    return clusters

# ─────────────────────────── Intra-Cluster Corr ─────────────────────────────
def intra_cluster_corr(rho_hat, clusters):
    """Average pairwise intra-cluster correlation (Eq. 9 in paper)."""
    total = 0.0
    count = 0
    for cl in clusters:
        n = len(cl)
        for ii in range(n):
            for jj in range(ii + 1, n):
                total += rho_hat[cl[ii], cl[jj]]
                count += 1
    return total / count if count > 0 else -np.inf

# ─────────────────────────── ACC Algorithm ──────────────────────────────────
def acc_algorithm(X, verbose=True):
    """
    Full ACC (Asset Clustering through Correlation) algorithm.
    Implements Algorithm 1 from the paper.
    """
    n, d = X.shape
    if verbose:
        print(f"\n{'='*55}")
        print(f"ACC  |  n={n} days, d={d} assets")
        print(f"{'='*55}")

    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    rho_hat = (X_std.T @ X_std) / (n - 1)
    np.fill_diagonal(rho_hat, 1.0)

    if verbose:
        print("Computing CORD matrix ...")
    cord = compute_cord(rho_hat)

    # ε search range: empirical CORD quantiles
    cord_upper = cord[np.triu_indices_from(cord, k=1)]
    q10, q50, q90 = np.percentile(cord_upper, [10, 50, 90])
    eps_min = max(PARAM_EPS_MIN, q10 * 0.5)
    eps_max = min(PARAM_EPS_MAX, q90 * 1.5)
    eps_grid = np.linspace(eps_min, eps_max, PARAM_NG)

    if verbose:
        print(f"  CORD quantiles: q10={q10:.4f}, q50={q50:.4f}, q90={q90:.4f}")
        print(f"  ε grid: [{eps_min:.4f}, {eps_max:.4f}] × {PARAM_NG} pts")

    # Grid search: Rule 3 (cluster count) + Rule 2 (max intra-cluster corr)
    best = dict(eps=None, avg_corr=-np.inf, num_k=None, clusters=None)

    for eps in eps_grid:
        clusters_eps = partition_procedure(cord, eps)
        num_k = len(clusters_eps)

        if not (PARAM_K_MIN <= num_k <= PARAM_K_MAX):
            continue

        avg_corr = intra_cluster_corr(rho_hat, clusters_eps)
        if np.isnan(avg_corr):
            continue

        if avg_corr > best["avg_corr"]:
            best = dict(eps=eps, avg_corr=avg_corr, num_k=num_k, clusters=clusters_eps)

    # Fallback: relax cluster range
    if best["eps"] is None:
        if verbose:
            print("  No ε in range — taking best available")
        for eps in eps_grid:
            clusters_eps = partition_procedure(cord, eps)
            avg_corr = intra_cluster_corr(rho_hat, clusters_eps)
            if not np.isnan(avg_corr) and avg_corr > best["avg_corr"]:
                best = dict(eps=eps, avg_corr=avg_corr, num_k=len(clusters_eps),
                            clusters=clusters_eps)

    clusters = best["clusters"] or []

    if verbose:
        print(f"\n  Best ε = {best['eps']:.4f}, Clusters = {len(clusters)}, "
              f"Intra-cluster ρ = {best['avg_corr']:.4f}")

    return clusters, dict(
        best_epsilon=best["eps"],
        num_clusters=len(clusters),
        intra_cluster_corr=best["avg_corr"],
        d=d, n=n,
        eps_range=(float(eps_min), float(eps_max)),
    )

# ─────────────────────────── Asset Selection ────────────────────────────────
def select_lowest_variance(X, clusters):
    """
    Theorem 2: for minimum variance, pick the asset with lowest variance
    from each cluster.
    """
    variances = np.var(X, axis=0)
    return [
        {"asset_index": int(min(cl, key=lambda j: variances[j])),
         "variance": float(min(variances[j] for j in cl)),
         "cluster_size": len(cl)}
        for cl in clusters
    ]

# ─────────────────────────── Single Analysis ────────────────────────────────
def run_single(returns_df, window=PARAM_WINDOW):
    n = min(window, len(returns_df))
    rets = returns_df.iloc[-n:]
    tickers = rets.columns.tolist()

    nan_frac = rets.isnull().mean()
    valid = nan_frac < 0.05
    rets = rets.loc[:, valid]
    tickers = [t for t, v in zip(tickers, valid) if v]
    X = np.nan_to_num(rets.values, nan=0.0)

    if X.shape[0] <= X.shape[1]:
        drop = X.shape[1] - X.shape[0] + 10
        keep = np.random.choice(X.shape[1], X.shape[1] - drop, replace=False)
        X = X[:, keep]
        tickers = [tickers[i] for i in keep]

    print(f"\nWindow: {n} days × {X.shape[1]} assets  "
          f"({rets.index[0].date()} → {rets.index[-1].date()})")

    clusters, metrics = acc_algorithm(X, verbose=True)
    selection = select_lowest_variance(X, clusters)

    sel_tickers = [tickers[s["asset_index"]] for s in selection]
    cluster_tickers = [sorted([tickers[j] for j in cl]) for cl in clusters]

    sel_rets = rets[sel_tickers].iloc[-n:]
    ann_ret = sel_rets.mean() * 252
    ann_vol = sel_rets.std() * np.sqrt(252)
    sharpe = (ann_ret / ann_vol).replace([np.inf, -np.inf], np.nan).mean()

    print(f"\n  Selected {len(sel_tickers)} stocks, Sharpe={sharpe:.3f}")

    return dict(
        metrics=metrics, selection=selection,
        cluster_tickers=cluster_tickers,
        tickers=tickers, sel_tickers=sel_tickers,
        ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
        window_start=str(rets.index[0].date()),
        window_end=str(rets.index[-1].date()),
        n=n, d=X.shape[1],
    )

# ─────────────────────────── Walk-Forward ───────────────────────────────────
def run_walkforward(returns_df, window=PARAM_WINDOW, step=126, n_snapshots=4):
    dates = returns_df.index.tolist()
    results = []
    starts = list(range(0, len(dates) - window, step))[:n_snapshots]

    for start_i in starts:
        end_i = start_i + window
        rets = returns_df.iloc[start_i:end_i]
        nan_frac = rets.isnull().mean()
        valid = nan_frac < 0.05
        rets_v = rets.loc[:, valid]
        tickers_v = [t for t, v in zip(rets.columns.tolist(), valid) if v]
        X = np.nan_to_num(rets_v.values, nan=0.0)

        if X.shape[0] <= X.shape[1]:
            drop = X.shape[1] - X.shape[0] + 5
            keep = np.random.choice(X.shape[1], X.shape[1] - drop, replace=False)
            X = X[:, keep]
            tickers_v = [tickers_v[i] for i in keep]

        try:
            clusters, metrics = acc_algorithm(X, verbose=False)
            sel = select_lowest_variance(X, clusters)
            sel_tickers = [tickers_v[s["asset_index"]] for s in sel]
            sd, ed = dates[start_i], dates[end_i - 1]
            results.append({
                "window_start": str(sd.date()) if hasattr(sd, "date") else str(sd),
                "window_end": str(ed.date()) if hasattr(ed, "date") else str(ed),
                "num_clusters": metrics["num_clusters"],
                "epsilon": float(metrics["best_epsilon"]),
                "intra_corr": float(metrics["intra_cluster_corr"]),
                "selected_tickers": sel_tickers,
            })
            print(f"  {results[-1]['window_start']} → {results[-1]['window_end']}: "
                  f"K={metrics['num_clusters']}, ε={metrics['best_epsilon']:.4f}, "
                  f"selected {len(sel_tickers)}")
        except Exception as e:
            print(f"  Window error: {e}")
    return results

# ─────────────────────────── Main ────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("ACC Algorithm  |  arXiv:2103.14506")
    print("=" * 60)

    returns_df = load_returns()

    print("\n" + "=" * 60)
    print("SINGLE-SHOT ACC ANALYSIS")
    print("=" * 60)
    result = run_single(returns_df, window=PARAM_WINDOW)

    print("\n--- Selected Stocks (lowest variance per cluster) ---")
    for cid, s in enumerate(result["selection"]):
        ticker = result["sel_tickers"][cid]
        print(f"  Cluster {cid:2d}: {ticker:<6s}  "
              f"var={s['variance']:.6f}  size={s['cluster_size']}")

    print("\n--- Cluster Composition ---")
    for cid, cl_tickers in enumerate(result["cluster_tickers"]):
        shown = ", ".join(cl_tickers[:10])
        if len(cl_tickers) > 10:
            shown += " ..."
        print(f"  Cluster {cid:2d} ({len(cl_tickers):3d}): {shown}")

    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY")
    print("=" * 60)
    wf = run_walkforward(returns_df, window=PARAM_WINDOW, step=126, n_snapshots=4)

    # ── Write output
    lines = []
    lines.append("=" * 70)
    lines.append("ACC Algorithm Output  |  arXiv:2103.14506")
    lines.append("Tang, Xu, Zhou — Asset Selection via Correlation Blockmodel Clustering")
    lines.append("=" * 70)
    lines.append("")
    lines.append("PARAMETERS")
    lines.append(f"  Lookback window : {PARAM_WINDOW} trading days")
    lines.append(f"  Cluster range  : [{PARAM_K_MIN}, {PARAM_K_MAX}] (Rule 3)")
    lines.append(f"  ε grid size    : {PARAM_NG}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("SINGLE-SHOT ACC RESULTS")
    lines.append("-" * 70)
    lines.append(f"Window           : {result['window_start']} → {result['window_end']}")
    lines.append(f"Assets analyzed  : {result['d']}")
    lines.append(f"Periods (days)   : {result['n']}")
    lines.append(f"Clusters found   : {result['metrics']['num_clusters']}")
    lines.append(f"Intra-cluster ρ  : {result['metrics']['intra_cluster_corr']:.4f}")
    lines.append(f"Best epsilon     : {result['metrics']['best_epsilon']:.6f}")
    lines.append("")
    lines.append("SELECTED STOCKS (Theorem 2: lowest variance per cluster):")
    for cid, s in enumerate(result["selection"]):
        ticker = result["sel_tickers"][cid]
        lines.append(f"  Cluster {cid:2d}: {ticker:<6s}  "
                     f"var={s['variance']:.6f}  size={s['cluster_size']}")
    lines.append("")
    lines.append("CLUSTER COMPOSITION:")
    for cid, cl_tickers in enumerate(result["cluster_tickers"]):
        lines.append(f"  Cluster {cid:2d} ({len(cl_tickers):3d} stocks): "
                     f"{', '.join(cl_tickers[:20])}"
                     f"{'...' if len(cl_tickers) > 20 else ''}")
    lines.append("")
    lines.append("PORTFOLIO METRICS (equal-weight, on selected stocks):")
    for t in result["sel_tickers"]:
        r = result["ann_ret"][t]
        v = result["ann_vol"][t]
        sr = (r / v) if v > 0 else 0.0
        lines.append(f"  {t:<6s}  ann_ret={r:+.2%}  ann_vol={v:.2%}  Sharpe={sr:+.2f}")
    lines.append("")
    lines.append(f"  Portfolio avg Sharpe ratio : {result['sharpe']:.3f}")

    if wf:
        lines.append("")
        lines.append("-" * 70)
        lines.append("WALK-FORWARD ACC RESULTS")
        lines.append("-" * 70)
        for r in wf:
            lines.append(f"  {r['window_start']} → {r['window_end']}: "
                         f"K={r['num_clusters']}, "
                         f"ε={r['epsilon']:.4f}, intra_corr={r['intra_corr']:.4f}")
            lines.append(f"    Selected: {', '.join(r['selected_tickers'])}")

    lines.append("")
    lines.append(f"Total runtime: {time.time() - t0:.1f}s")

    output_text = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(output_text)

    print(f"\nOutput written to: {OUT_FILE}")
    print(f"Runtime: {time.time() - t0:.1f}s")
    print("Done!")
