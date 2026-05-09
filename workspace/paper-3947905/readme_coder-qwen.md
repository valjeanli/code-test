# Paper Summary: Non-parametric Online Market Regime Detection (SSRN 3947905 / arXiv 2306.15835)

## Metadata
- **Title**: Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures
- **Authors**: Blanka Horvath (Oxford, Man Institute, Turing Institute) & Zacharia Issa (King's College London)
- **Year**: 2023
- **Venue**: SSRN / arXiv preprint

---

## Problem Statement
Financial time series are non-stationary and exhibit heteroscedasticity. Market regimes are periods where the underlying distribution generating asset price dynamics remains stable. The paper addresses two problems:
1. **MRDP (Market Regime Detection Problem)**: Online detection of when the underlying distribution changes
2. **MRCP (Market Regime Clustering Problem)**: Grouping historical periods into clusters of similar market activity

The key innovation: using path signatures + Maximum Mean Discrepancy (MMD) as a non-parametric two-sample test that works on path space (not just return distributions).

---

## Core Methodology

### 1. Path Signatures
The signature of a path X over [a,b] is the infinite sequence of iterated integrals:

S(X) = (1, X¹, X², ..., Xᴺ, ...)

where Xᵏ = ∫...∫ dX⊗ᵏ (k-fold iterated integral, Eq. 6).

Key properties:
- Uniqueness up to tree-like equivalence (recovers path distribution)
- Time-reparameterization invariant
- Universal approximator of continuous path functionals

The paper uses the **signature kernel** (rather than truncated signatures) via the kernel trick to avoid the curse of dimensionality (Section 2.2).

### 2. Signature Maximum Mean Discrepancy (sig-MMD)
The MMD between distributions P and Q on path space using the signature kernel k_sig:

MMD²(P,Q) = E[k(X,X')] + E[k(Y,Y')] - 2·E[k(X,Y)]

The **unbiased estimator** (Eq. 16):

D²_sig = (1/(n(n-1))) Σᵢ≠ⱼ k(xᵢ,xⱼ) + (1/(m(m-1))) Σᵢ≠ⱼ k(yᵢ,yⱼ) - (2/(nm)) Σᵢⱼ k(xᵢ,yⱼ)

### 3. Online Regime Detection Pipeline (Section 4)

**Step 1: Partition the data stream**
- Slice the time-augmented price series into non-overlapping sub-paths of length h₁
- Group consecutive sub-paths into ensembles of size h₂
- Hyperparameters: h = (h₁, h₂). Paper uses h=(7,10) for hourly data, h=(8,16) for daily

**Step 2: Path transformations**
- φ_norm: Normalize the path (e.g., scale)
- φ_time: Add time as a monotone channel
- φ_incr: Convert prices to increments (returns)
- Φ = φ_incr ∘ φ_time ∘ φ_norm

**Step 3: Compute sig-MMD scores**
- For each ensemble, compute the sig-MMD distance against a reference distribution
- **Parametric beliefs**: compare against simulated paths from a model (e.g., GBM with known parameters)
- **Auto-evaluator (non-parametric)**: compare each ensemble against the previous lag (self-comparison)

**Step 4: Threshold determination**
- Bootstrap the null distribution: draw many ensemble pairs from the reference, compute sig-MMD
- Set critical value c_α at the (1-α) quantile (e.g., α=0.05)
- If MMD score > c_α → regime change detected

**Step 5: Auto-evaluator (no parametric beliefs needed)**
- Compare ensemble at time t against ensemble at time t-ℓ (lag ℓ)
- Multiple lags can be averaged for smoothing
- This is a fully non-parametric approach

### 4. Regime Clustering (Section 5)
- Use MMD distances between ensembles as input to agglomerative hierarchical clustering
- Alternative: sliced Wasserstein k-means on signature features
- Produces a regime label per time period

---

## Pseudocode

```
Input: Time-augmented price series s_hat = [(t, p₁,...,p_d), ...]
Params: h₁ (sub-path length), h₂ (ensemble size), sig_kernel params

# 1. Partition into sub-paths
sub_paths = [s_hat[j*h₁ : (j+1)*h₁] for j in 0..N₁]

# 2. Form ensembles  
ensembles = [(sub_paths[k], ..., sub_paths[k+h₂-1]) for k in 0..N₂]

# 3. Transform each ensemble
transformed = [apply_transformations(ens) for ens in ensembles]  # norm, time, incr

# 4. Compute pairwise sig-MMD distances
for each ensemble e_i:
    # Auto-evaluator: compare with lag ℓ
    ref_ensemble = transformed[i-ℓ]  if i≥ℓ else None
    if ref_ensemble exists:
        mmd_score[i] = sig_mmd(reference_paths, e_i)
    
    # Parametric mode: compare with simulated beliefs
    for each belief B_j:
        mmd_score[j][i] = sig_mmd(simulated_paths(B_j), e_i)

# 5. Threshold and detect
critical_value = bootstrap_quantile(sig_mmd, reference, alpha=0.05)
change_points = [i where mmd_score[i] > critical_value]
```

---

## Implementation Notes

### Signature Kernel Approximation
Computing the exact signature kernel is expensive. This implementation uses:
1. **Truncated signatures** up to level N (typically N=5-6) as a practical approximation
2. **RBF or linear kernel** applied to the flattened signature feature vectors
3. The truncated signature of a d-dimensional path at level N has O(dᴺ) terms — manageable for d≤5

### Practical Simplifications
- Single asset (S&P 500 or individual stock) → d=1 price + 1 time channel
- Daily data → sub-path length h₁ = 20 (≈1 month), ensemble size h₂ = 10
- Auto-evaluator with lag ℓ=1 (compare to immediately preceding period)
- MMD computed using pre-computed kernel matrices (O(n²m²) but small n,m)

### Computational Complexity
- Signature computation: O(N·h₁·d·level!) per path
- MMD: O(n²) kernel evaluations for n samples
- Overall: O(N·h₁·d·level! + n²) per sliding window step

---

## Hyperparameters Used in This Implementation
- h₁ = 20 (sub-path length, ~1 month of daily data)
- h₂ = 10 (ensemble size)
- signature_level = 5
- lag = 1 (auto-evaluator compares to previous ensemble)
- alpha = 0.05 (significance level)
- n_bootstrap = 200 (bootstrap samples for threshold)
- RBF kernel gamma = 1.0 / n_features

---

## Evaluation

### Scores

| Metric | Score (out of max) | Notes |
|--------|-------------------|-------|
| Readme Quality | 6/10 | Decent explanation of paper methodology — covers path signatures, sig-MMD, online detection pipeline, and pseudocode. Less detailed and lacks the depth of other implementations |
| Code Quality | 6/10 | 12KB code analyzing 10 individual stocks (ADBE, ABBV, AMD, etc.) rather than S&P 500. Implements MMD signature-based detection with multi-stock support. Uses truncated signatures with RBF kernel |
| Output Compliance | 1/10 | Produces CSV and metadata JSON but with **severe violations**: `predicted_regime` starts at 10 (not 0), labels are `regime_10` through `regime_15` (not standard), only 57 rows (monthly samples, not daily), and `num_regimes=6` but labels map has keys 10-15. Only 6 ensemble points per stock |
| Output Quality | 3/10 | Multi-stock analysis is an interesting approach but **does not produce S&P 500 regime detection**. Only 11 total change points across 10 stocks with 6 data points each. The sparse sampling (monthly ensembles) misses most regime dynamics |
| Adherence to T2 Spec | 1/10 | Severely non-compliant — wrong asset (individual stocks vs S&P 500 index), wrong date sampling, non-standard regime labels, small number of output rows |
| **Total** | **17/50** | |

### Strengths
- Multi-stock analysis approach is novel — tests regime detection across different equities
- Produces output files in the right format (CSV + JSON) even if data is non-compliant
- MMD-based detection with proper kernel computation

### Weaknesses
- **Analysis is on individual stocks, not S&P 500 as required**
- **`predicted_regime` values start at 10 instead of 0** — violates the spec's contiguous integer constraint
- Only 57 output rows (6 per stock) instead of ~1260 daily rows for the 5-year window
- Extremely sparse temporal sampling — 3-4 month gaps between data points
- Only 6 ensemble points per stock cannot produce meaningful regime detection
