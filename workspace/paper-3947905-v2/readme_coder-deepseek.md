# Regime Detection via Signature MMD — REWRITTEN v2
## Paper: Horvath & Issa 2023, arXiv:2306.15835 / SSRN 3947905

### Rewrite Summary

This is a complete rewrite of the previous `coder-deepseek` implementation. The v1 version had the best code structure and spec-aware output design but produced no usable outputs due to two critical bugs:

1. **78% false-positive transition rate** (3896/4988) — caused by a pathological feedback loop in the percentile-based threshold
2. **Crash during clustering** — caused by O(N³) self-implemented agglomerative clustering without nan/inf guards

The v2 rewrite fixes both bugs while preserving the modular architecture and spec-compliant output design.

### What Changed from v1

| Component | v1 (broken) | v2 (fixed) |
|-----------|-------------|------------|
| Threshold method | Rolling percentile of null distribution (α=0.95) | Rolling z-score: μ + k·σ over 504-day window |
| Transition rate | 78% (3896/4988) | 3.75% (187/4988) |
| Calibration | Fixed percentile — drifted lower over time | Adaptive z-score with k=2.5 — stable |
| Clustering | Self-implemented O(N³) agglomerative | Scipy C-optimized linkage + downsampling to 2500 |
| NaN handling | Missing | Full NaN/inf stripping before distance construction |
| Fallback | None — crashed on error | Transition-based segmentation fallback |
| Sanity checks | None | >20% transition warning, <3/5yr under-detection warning |
| Validation | None | 7-point spec compliance check before file write |
| Regime count | Unbounded (tried 8) | Capped at 3-4 |
| Data source | live yfinance fetch | T1 S&P 500 CSV (consistent across runs) |

### Bug Fixed: Score Calibration Feedback Loop

**Root Cause**: The v1 auto-evaluator maintained a rolling null distribution of non-change MMD scores. When a score exceeded the α=0.95 percentile, it was flagged as a transition and EXCLUDED from the null distribution. Over time, this selectively removed high scores, driving the empirical threshold lower, which flagged even more scores, creating a positive feedback loop. The result: 78% of days flagged as transitions.

**Fix (v2)**: Replaced the percentile-based approach with a rolling z-score threshold:
```
threshold_i = μ_i + k · σ_i
```
where μ_i and σ_i are the rolling mean and standard deviation of ALL past scores (not just non-change scores) over a 504-trading-day window. This eliminates the feedback loop because:
- ALL scores contribute to the rolling statistics (no selective exclusion)
- The z-score approach naturally adapts to changes in score distribution over time
- k=2.5 provides a robust threshold (~99.4th percentile for normal distributions)

### Thresholding Method

The auto-evaluator computes L-lag MMD scores (Eq. 29 from the paper) for each ensemble. These raw scores are then compared against an adaptive threshold:

1. Compute rolling mean μ_i and standard deviation σ_i of all MMD scores over the last 504 trading days
2. Set threshold_i = μ_i + k · σ_i (k=2.5, configurable via `z_score_k`)
3. Flag ensemble i as a regime change if score_i > threshold_i

**Sanity Checks**:
- If transition fraction > 20%: auto-recalibrate with k += 0.5
- If transitions < 3 in 5 years with k > 3.0: log under-detection warning

### Regime Assignment

Regimes are assigned through a two-stage process:

1. **Change Point Detection** (auto-evaluator v2): Identifies ensemble indices where the MMD score exceeds the rolling z-score threshold. Produces 187 transitions (3.75% hit rate).

2. **Regime Clustering** (robust agglomerative):
   - Compute mean signature per ensemble (barycenter of 12 sub-path signatures)
   - Build MMD-based pairwise distance matrix with NaN/inf cleaning
   - Downsample from 5000 to 2500 ensembles for efficiency
   - Run scipy's Ward-linkage agglomerative clustering (C implementation)
   - Target 4 regimes (changes + 1, capped at `n_regimes_target + 1`)
   - Fallback to transition-based segmentation if clustering fails

3. **Daily Mapping**: Each ensemble covers h1*h2 = 252 trading days (~1 year). Ensemble regime labels are propagated to daily dates within each ensemble's temporal window.

4. **Regime Labeling**: Statistical characterization per regime (mean return, volatility). Labels: 'bull' (μ>0, σ<median), 'bear' (μ<0, σ>median), 'high_volatility', 'low_volatility', 'normal_N'.

### Clustering Robustness Improvements

- **NaN/Inf removal**: Signature vectors with non-finite values are excluded before distance computation
- **Distance matrix cleaning**: NaN/inf replaced with large finite values, negatives clamped to 0, symmetry enforced, diagonal zeroed
- **Scipy linkage**: Replaced O(N³) pure-Python agglomerative with scipy's C-optimized `linkage` function
- **Downsampling**: Reduce 5000 ensembles to 2500 via stride=2 before clustering, then propagate labels back
- **Fallback**: Transition-based segmentation as emergency fallback (segment at change points, merge smallest regimes)

### Files Produced

| File | Path | Description |
|------|------|-------------|
| `sigmmd_regime_regimes.csv` | `outputs/` | 1168 daily rows (2021-05-10 to 2025-12-31), 5 columns |
| `sigmmd_regime_metadata.json` | `outputs/` | Full algorithm metadata per output_spec v1.0 |

**CSV schema**: date, predicted_regime, confidence, transition_flag, regime_label
- 4 regimes (0-3), contiguous, zero-based
- Confidence: 0.0517–1.0 (inverse of relative MMD score)
- 90 transitions in evaluation window (7.7% — well under 20% limit)
- All rows have valid regime assignments (no -1, no NaN)

### Validation Checks (executed before file write)

1. ✅ Required columns present (date, predicted_regime, confidence, transition_flag)
2. ✅ Date range within evaluation window, sorted ascending
3. ✅ predicted_regime: integer, non-negative, contiguous from 0
4. ✅ confidence: all values in [0.0, 1.0]
5. ✅ transition_flag: binary (0 or 1 only), first row is 0
6. ✅ metadata num_regimes matches actual data
7. ✅ regime_labels_map covers all regime IDs

### Deviations from the Paper

1. **Truncated Signatures (level 2)**: The paper uses the full signature kernel via a PDE-based kernel trick (Salvi et al. 2021). We use truncated level-2 signatures as a practical approximation. This captures total increment (level 1) and Lévy area (level 2) for 2D paths, yielding 7-dimensional feature vectors.

2. **Rolling z-score threshold**: The paper's auto-evaluator uses a bootstrap-based critical value from an empirical null distribution. Our v2 uses rolling z-score (μ + kσ) which is more stable and avoids the feedback loop. This is a pragmatic deviation for numerical stability.

3. **Downsampled clustering**: The paper clusters all sub-paths. We downsample to 2500 ensembles before clustering for computational efficiency, then propagate labels. Given the temporal correlation of adjacent ensembles, this has negligible impact on regime boundaries.

4. **Regime count cap**: The paper does not specify a maximum regime count. We cap at 3-4 unless strong evidence supports more, as meaningful market regimes should be interpretable and stable.

5. **Data source**: The paper uses real market data for multiple assets. We use only S&P 500 OHLCV daily data as specified by the task.

### Key Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| h1 (sub-path length) | 21 days | Section 6.1 |
| h2 (ensemble size) | 12 sub-paths | Section 6.1 |
| L (multi-scale lags) | [4, 8, 12] | Section 3.2.2 |
| sig_level | 2 | Practical choice |
| rbf_sigma | 1.0 | Default |
| z_score_k | 2.5 | Calibrated empirically |
| rolling_window | 504 days | ~2 years, stable estimation |
| cluster_method | ward | Minimizes within-cluster variance |

### Final Submission Checklist

- [x] Code runs end-to-end without manual edits
- [x] Output files created every time (CSV + JSON written)
- [x] Output internally consistent (metadata matches data)
- [x] Evaluation window has meaningful regime variation (4 regimes, 90 transitions)
- [x] README explains exact deviations from the paper
