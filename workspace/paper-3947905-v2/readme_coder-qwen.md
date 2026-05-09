# Regime Detection: Horvath & Issa (2023) — Rewrite v2.1

## Metadata
- **Paper**: Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures
- **Authors**: Blanka Horvath & Zacharia Issa
- **Reference**: SSRN 3947905 / arXiv:2306.15835
- **Implementation**: coder-qwen, rewrite v2.1

---

## What Changed from v1

| Issue in v1 | Fix in v2.1 |
|---|---|
| Analyzed 10 individual stocks (ADBE, ABBV, AMD, etc.) | Single asset: S&P 500 index only |
| Only 57 output rows (6 per stock, monthly ensemble sampling) | 1168 daily rows covering full evaluation window |
| Regime IDs started at 10 (non-contiguous, not zero-based) | Regime IDs are 0, 1, 2 (contiguous, zero-based) |
| No CSV/JSON output files | Canonical `sig_mmd_regime_regimes.csv` + `sig_mmd_regime_metadata.json` |
| Multi-stock structure in output | Single-asset output pipeline |

## Bugs Fixed

1. **Wrong asset**: The previous version filtered by individual stock symbols instead of using the S&P 500 index data directly. The parquet file contains the S&P 500 index OHLCV (columns: date, open, high, low, close, adj_close, volume) — no symbol column needed.

2. **Sparse sampling**: The previous version only produced one row per ensemble (6 per stock), resulting in 57 rows total. The new version produces one row per trading day using rolling signature features.

3. **Regime ID indexing**: The previous version used raw k-means cluster IDs (10, 11, 12, etc.) without remapping. The new version explicitly remaps to contiguous zero-based integers [0, 1, 2].

4. **Missing output files**: The previous version wrote only a text summary. The new version writes spec-compliant CSV + JSON.

---

## Thresholding Method

The implementation uses two complementary approaches:

### 1. Auto-Evaluator MMD (Eq. 30, Def. 3.5) — Diagnostics
- Partition data into non-overlapping sub-paths of h1=20 days (Eq. 24)
- Group consecutive sub-paths into ensembles of h2=10 (Eq. 25)
- Compute sig-MMD between consecutive ensembles (auto-evaluator)
- Bootstrap threshold at alpha=0.05 significance level
- This produces diagnostic change points (not directly used for daily regime assignment)

### 2. Daily Feature Clustering (Sec. 5) — Primary Regime Assignment
- Build daily signature features using rolling windows of h1=20 days
- Each day t gets signature features from [t-h1+1, t]
- Apply k-means clustering (k=3) to daily feature vectors
- **Temporal smoothing**: majority-vote sliding window of 21 trading days to remove spurious single-day flips
- This produces stable, economically meaningful regime assignments

### Adaptive Threshold Logic
If the auto-evaluator detects zero transitions or an absurdly high transition rate (>50%), the threshold is automatically adjusted (halved or multiplied by 1.5) for up to 3 retries.

---

## Regime Assignment

Regimes are assigned via k-means clustering on daily signature features, with post-hoc temporal smoothing:

1. **Feature extraction**: For each day, compute truncated path signature (levels 1-2) from a 20-day rolling window of [close price, log return, volume ratio]
2. **Clustering**: k-means with k=3 clusters on the standardized signature features
3. **Smoothing**: Centered majority-vote filter with window=21 trading days removes noise
4. **Remapping**: Labels remapped to contiguous zero-based integers [0, 1, 2]
5. **Labels**: Regime 0 = bull, Regime 1 = bear, Regime 2 = sideways

Confidence scores are computed as 1 - (normalized distance to nearest centroid).

---

## Files Produced

| File | Path | Description |
|---|---|---|
| CSV | `sig_mmd_regime_regimes.csv` | 1168 rows, one per trading day |
| JSON | `sig_mmd_regime_metadata.json` | Algorithm metadata, parameters, regime labels |
| Code | `code_coder-qwen.py` | Full implementation |
| Output | `output_coder-qwen.txt` | stdout/stderr from execution |

---

## Validation Checks

The implementation validates output before writing:

- [x] Date range covers evaluation window (2021-05-10 to 2025-12-31; data ends 2025-12-31)
- [x] All required columns present (date, predicted_regime, confidence, transition_flag, regime_label)
- [x] predicted_regime is integer and non-negative
- [x] confidence is in [0, 1]
- [x] transition_flag is binary (0 or 1), first row = 0
- [x] Regime IDs are contiguous zero-based [0, 1, 2]
- [x] num_regimes in metadata matches max(predicted_regime) + 1
- [x] No NaN values in any column
- [x] Transition rate = 3.94% (well below 30% flicker threshold)
- [x] 1168 rows (header + data)

---

## Output Statistics

| Metric | Value |
|---|---|
| Trading days | 1168 |
| Date range | 2021-05-10 to 2025-12-31 |
| Regimes | 3 (0=bull, 1=bear, 2=sideways) |
| Regime 0 (bull) | 354 days (30%) |
| Regime 1 (bear) | 263 days (23%) |
| Regime 2 (sideways) | 551 days (47%) |
| Transitions | 46 |
| Transition rate | 3.94% |
| Avg dwell time | ~25 trading days |

---

## Deviations from the Paper

1. **Daily vs. ensemble granularity**: The paper's algorithm operates at ensemble level (comparing groups of sub-paths). For daily output, we use rolling windows to compute daily signature features, then cluster at daily resolution with temporal smoothing. The ensemble MMD is still computed for diagnostics (Eq. 30).

2. **Single-asset vs. multidimensional**: The paper targets multidimensional portfolios. This implementation uses S&P 500 index only (3 channels: price, log return, volume ratio).

3. **k-means vs. sliced Wasserstein**: The paper mentions sliced Wasserstein k-means as an alternative. We use standard k-means with RBF-like distance preprocessing via signature features.

4. **Temporal smoothing**: The paper does not specify post-clustering smoothing. We add a 21-day majority-vote filter to ensure economically meaningful regime persistence (avoiding the "flicker" problem).

5. **Fallback handling**: If k-means fails, we fall back to quantile-based assignment rather than crashing.

---

## Hyperparameters

| Parameter | Value | Source |
|---|---|---|
| h1 (sub-path length) | 20 | Paper Section 4 (daily data) |
| h2 (ensemble size) | 10 | Paper Section 4 |
| signature_level | 2 | Practical truncation |
| alpha | 0.05 | Def. 3.5 significance level |
| n_bootstrap | 500 | Bootstrap samples |
| n_regimes | 3 | Clustering parameter |
| smooth_window | 21 | Temporal smoothing (1 trading month) |
| random_seed | 42 | Reproducibility |

---

## Dependencies

- Python 3.13+
- numpy, pandas, pyarrow (data processing)
- scikit-learn (k-means clustering)
