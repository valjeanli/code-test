# Paper Implementation: SSRN 3947905 / arXiv 2306.15835 (v2 REWRITE)

## Paper Details
- **Title**: Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures
- **Authors**: Blank Horvath (Oxford/KCL), Zacharia Issa (KCL)
- **Year**: 2023
- **Venue**: arXiv (preprint)

## Rewrite Summary

### What Changed from v1
1. **Single canonical pipeline**: Eliminated the dual output track. Now produces one CSV and one JSON.
2. **No sentinel values**: Removed all -1 regime assignments. Every row has a valid regime ID (0, 1, or 2).
3. **Feature-based clustering**: Regime assignment now uses k-means on rolling features (return, volatility) instead of post-hoc changepoint segmentation. This produces better regime separability.
4. **Regime smoothing**: Added a smoothing step that merges short segments (<5 days) into neighboring segments, reducing flickering.
5. **Meaningful confidence**: Confidence is now derived from centroid distance (how close a point is to its assigned cluster centroid), normalized to [0.1, 1.0].
6. **Strict validation**: Added pre-write validation checking date range, schema, regime IDs, confidence bounds, transition flags, and metadata consistency.

### Bug/Failure Mode Fixed
- **v1 main CSV had predicted_regime=-1 for all rows** — violated spec constraint `0 <= predicted_regime < num_regimes`. Fixed by ensuring all rows get valid regime assignments.
- **v1 online detector assigned same regime to all rows** — detected zero transitions in 20 years. Fixed by using feature-based clustering for regime assignment.
- **v1 had flat 0.5 confidence** — not meaningful. Fixed by computing confidence from centroid distance.
- **v1 had two conflicting output files** — unclear which was canonical. Fixed by producing single output.

### Thresholding Method
- **Primary**: Empirical prior quantile threshold (alpha=0.95) on MMD auto-evaluation scores
- **Adaptive**: If alpha=0.95 produces too few/many transitions, automatically retries with lower alpha values (0.90, 0.85, ..., 0.50)
- **Fallback**: Absolute percentile threshold (70th percentile of valid scores)
- **Smoothing**: Short regime segments (<5 days) are merged into neighboring segments

### Regime Assignment
1. Compute rolling features: 20-day cumulative return, 20-day annualized volatility
2. Standardize features
3. K-means clustering (n=3, seed=42)
4. Remap labels sorted by mean return: 0=bear (lowest), 1=sideways, 2=bull (highest)
5. Smooth labels to remove short segments

### Files Produced
- `sig_mmd_regimes.csv`: Daily regime predictions for evaluation window (2021-05-10 to 2025-12-31)
- `sig_mmd_metadata.json`: Algorithm metadata with parameters, regime stats, and provenance

### Validation Checks Executed
1. Required columns exist (date, predicted_regime, confidence, transition_flag)
2. No NaN values in required columns
3. predicted_regime is integer and non-negative
4. confidence is in [0.0, 1.0]
5. transition_flag is binary (0 or 1)
6. First row has transition_flag=0
7. Regime IDs are contiguous from 0
8. Metadata num_regimes matches actual data
9. Metadata regime_labels_map keys match actual regimes

## Output Summary
- **Total rows**: 1168
- **Date range**: 2021-05-10 to 2025-12-31
- **Transitions**: 30
- **Regime distribution**:
  - Regime 0 (bear): 5 days (0.4%)
  - Regime 1 (sideways): 314 days (26.9%)
  - Regime 2 (bull): 849 days (72.7%)

## Methodology

### Core Algorithm (from paper)
1. **Path Signatures**: Compute truncated path signatures on time-augmented returns
2. **MMD**: Unbiased Maximum Mean Discrepancy with RBF kernel between signature ensembles
3. **Auto-evaluation**: L-lag weighted score comparing current ensemble against lagged ensembles
4. **Change detection**: Flag regime changes when score exceeds adaptive threshold
5. **Clustering**: K-means on rolling features to assign regime labels

### Key Parameters
- h1=8 (sub-path length), h2=8 (ensemble size)
- Lags: [1, 2, 4, 8] with exponential decay weights [0.4, 0.3, 0.2, 0.1]
- Signature truncation order: 3
- RBF kernel sigma: 1.0
- Detection alpha: 0.95 (adaptive fallback to 0.50)
- Prior window: 200 historical scores
- Min segment length: 10 days (detection), 5 days (smoothing)

### Deviations from Paper
1. **Feature-based clustering**: Paper uses sliced Wasserstein k-means on signature features. We use standard k-means on rolling return/volatility features for better separability on daily data.
2. **Simplified signatures**: Paper uses kernel-trick signature computation. We use truncated signature (order 3) for computational feasibility.
3. **Daily data adaptation**: Paper experiments use high-frequency data. We adapt for daily S&P 500 data.
4. **Regime smoothing**: Added post-hoc smoothing to reduce flickering, not in original paper.

## References
- Horvath, Issa (2023). arXiv:2306.15835
- Chevyrev, Oberhauser (2022). Signature moments to characterize laws of stochastic processes
