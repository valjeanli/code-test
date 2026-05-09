# Non-parametric Online Market Regime Detection (v2 Rewrite)

## Paper: SSRN 3947905 / arXiv 2306.15835

**Authors**: Blanka Horvath (Oxford), Zacharia Issa (King's College London)
**Date**: June 2023
**Venue**: arXiv preprint

---

## What Changed from v1

### Bug Fixed
- **Wrong evaluation window**: v1 used 2010-2020 data from Yahoo Finance instead of the required 2021-2025 S&P 500 OHLCV from the shared parquet file. v2 correctly uses the evaluation window 2021-05-10 through 2025-12-31 (end of available data).
- **No output files produced**: v1 generated only a text report with no CSV or JSON output. v2 produces spec-compliant `{algo_name}_regimes.csv` and `{algo_name}_metadata.json`.

### What Changed
1. **Data pipeline rebuilt** around the correct target period, using the shared `sp500_ohlcv_20yr.parquet` data.
2. **Added main() entry point** that loads S&P 500 data, computes features and regime scores, assigns regimes for the evaluation window, writes canonical CSV + metadata JSON, and runs validation.
3. **Added post-processing pipeline**: raw MMD change points are mapped to daily regime labels, short segments (< 15 days) are merged, and raw regimes are clustered into 4 target regimes using K-means on mean-return/volatility profiles.
4. **Machine-readable output**: replaces text-style reporting with CSV and JSON files conforming to the output spec.
5. **Validation checks** run before final write: date range, required columns, regime ID contiguity, confidence bounds, transition flag validity, metadata consistency.

---

## Thresholding Method

The auto-evaluator (Algorithm in Sec 3.2 of the paper) computes L-lag MMD scores over sliding windows of ensemble paths. At each time step, a **dynamic threshold** is computed by:

1. Collecting MMD scores from a sliding memory window (last 100 observations).
2. Fitting a **Gamma distribution** to the scores: shape α = mean²/var, scale β = var/mean.
3. Computing the critical threshold as the (1-α) quantile of the fitted Gamma distribution, where α = 0.05 (95% confidence).
4. If Gamma fitting fails (degenerate variance), falling back to the empirical 95th percentile.

Additionally, safety checks:
- If zero regime changes are detected (all scores below threshold), a top-5% percentile fallback is used.
- If more than 30% of ensembles are flagged as changes (excessive fragmentation), the threshold is tightened to the 99th percentile.

---

## Regime Assignment

### Step 1: Change-Point Detection
The signature-kernel MMD auto-evaluator identifies ensemble indices where MMD scores exceed the dynamic threshold. These are the change points.

### Step 2: Daily Mapping
Each change point (ensemble index) is mapped to a day index: `day = ensemble_index × h1`. Days between consecutive change points share the same regime ID.

### Step 3: Short Segment Merging
Any regime segment shorter than 15 trading days is merged into its longer neighboring segment.

### Step 4: K-Means Clustering
The raw change-point-based regime labels (potentially many) are clustered into 4 target regimes using K-means on the (mean_daily_return, daily_volatility) profile of each regime. This produces economically meaningful labels:
- **bull** (low vol, positive return)
- **calm** (moderate vol, positive return)
- **bear** (moderate vol, negative return)
- **crisis** (high vol, negative return)

### Step 5: Confidence Assignment
Confidence is computed based on distance from segment boundaries: highest in the middle of a regime, lowest near transitions. Range: [0.5, 1.0].

### Fallback
If the MMD detector finds only 1 regime (no meaningful variation), a volatility-based fallback is used: 20-day rolling volatility quantized into 3 regimes (bull/sideways/bear).

---

## Files Produced

| File | Description |
|------|-------------|
| `code_coder-glm.py` | Full implementation with main() entry point |
| `sig_mmd_horvath_issa_regimes.csv` | Daily regime predictions for evaluation window |
| `sig_mmd_horvath_issa_metadata.json` | Algorithm metadata, parameters, and provenance |
| `output_coder-glm.txt` | Captured stdout/stderr from running the code |

---

## Evaluation Window Output

- **Date range**: 2021-05-10 to 2025-12-31 (1168 trading days)
- **Regimes**: 4 (bear, crisis, bull, calm)
- **Transitions**: 5

| Regime | Label | Days | Mean Return | Daily Vol | Description |
|--------|-------|------|-------------|-----------|-------------|
| 0 | bear | 40 | -0.008%/d | 1.04%/d | Moderate-vol negative return |
| 1 | crisis | 280 | -0.078%/d | 1.64%/d | High-vol sustained drawdown |
| 2 | bull | 135 | +0.086%/d | 0.68%/d | Low-vol positive return |
| 3 | calm | 713 | +0.093%/d | 0.84%/d | Moderate-vol positive return |

Key detected transitions:
- 2021-05-10 → Nov 2021: bull phase
- 2021-11-18: brief bear (consolidation)
- 2022-01-18: crisis onset (Fed rate hike cycle)
- 2022-11-16: recovery/calm
- 2024-12-27: brief crisis (year-end volatility)
- 2025-04-10: return to calm

---

## Validation Checks Executed

Before writing output, the pipeline validates:

1. ✅ Date range starts at 2021-05-10
2. ✅ Required columns exist: date, predicted_regime, confidence, transition_flag
3. ✅ predicted_regime is integer, non-negative, contiguous 0-based (0,1,2,3)
4. ✅ confidence values are in [0.0, 1.0]
5. ✅ transition_flag is binary (0/1) with first row = 0
6. ✅ No NaN values in required columns
7. ✅ metadata.num_regimes matches max(predicted_regime)+1
8. ✅ regime_labels_map covers all regime integers

---

## Deviations from the Paper

1. **Truncated signatures** (level 2) instead of the full PDE-based signature kernel (Theorem 2.17). The paper's exact PDE solution is computationally infeasible for real financial data; our truncated approximation captures the essential path features.
2. **K-means post-clustering**: The paper defines regime change points but does not specify how to assign economically meaningful labels. We use K-means clustering on return/volatility profiles to map raw segments into interpretable regimes.
3. **RBF kernel on signatures**: We use an RBF kernel on truncated signature features rather than the exact signature kernel. This is a practical approximation that works well for daily financial data.
4. **Increment transform**: We include the increment (cumulative absolute change) channel as a preprocessing step, which the paper mentions but does not specifically recommend for all settings.
5. **Gamma distribution threshold with safety checks**: We add fallback thresholds for degenerate cases (zero changes, excessive changes) that the paper does not address.
6. **Evaluation window**: The paper tests on 1980-2023 US equities and 2019-2022 crypto. We test on S&P 500 2021-2025 only.

---

## Paper Summary (preserved from v1)

### Problem Statement
The paper addresses:
1. **MRDP** (Market Regime Detection Problem): Online detection of distributional changes in financial time series.
2. **MRCP** (Market Regime Clustering Problem): Grouping periods of similar market activity into distinct regimes.

### Core Algorithm
The signature kernel MMD uses iterated path integrals (truncated at level 2) to compare market path segments. The auto-evaluator computes L-lag scores by comparing each ensemble of sub-paths against lagged versions, then fits a Gamma distribution to recent scores for dynamic thresholding.

### Key Equations
- **MMD estimator**: D²_u(P,Q) = (1/n(n-1))Σ_{i≠j}k(x_i,x_j) - (2/mn)Σ_iΣ_jk(x_i,y_j) + (1/m(m-1))Σ_{i≠j}k(y_i,y_j)
- **L-lag score**: A_L[i] = Σ_{l∈L} w_l · D_sig(s_{i-l}, s_i)
- **Gamma threshold**: shape α = E[D]²/Var(D), scale β = Var(D)/E[D]
- **Signature kernel PDE**: ∂²f/∂s∂t = f(s,t)·⟨dX_s, dY_t⟩_V with f(0,t) = f(s,0) = 1