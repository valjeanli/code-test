# Non-parametric Online Market Regime Detection
## Paper: SSRN 3947905 / arXiv 2306.15835

**Authors**: Blanka Horvath (Oxford), Zacharia Issa (King's College London)  
**Date**: June 2023  
**Venue**: arXiv preprint

---

## 1. Problem Statement

The paper addresses two key problems:

1. **Market Regime Detection Problem (MRDP)**: Online detection of distributional changes in financial time series as data arrives in real-time.

2. **Market Regime Clustering Problem (MRCP)**: Grouping periods of similar market activity into distinct regimes (offline clustering).

Key challenges:
- Financial time series are non-stationary, heteroscedastic, and exhibit volatility clustering
- Need for real-time detection (online), not just retrospective analysis
- Must handle high-dimensional, path-dependent, non-Markovian data
- Traditional methods struggle with multivariate non-parametric setups

---

## 2. Core Methodology

### 2.1 Path Signatures

The **path signature** S(X) is a feature map on path space consisting of iterated integrals:

```
S(X) = (1, X^1, X^2, ...)
where X^k[a,b] = ∫_{a<u1<...<uk<b} dX_u1 ⊗ ... ⊗ dX_uk
```

**Key Properties**:
- **Uniqueness**: Signature is unique up to tree-like equivalence
- **Time reparametrization invariance**: Independent of sampling rate
- **Universality**: Linear functionals on signatures can approximate any continuous function on path space

### 2.2 Signature Kernel MMD

The **signature kernel** between two paths X, Y:

```
k_sig(X, Y) = ⟨S(X), S(Y)⟩_T((E))
```

The kernel can be computed via solving a PDE (Theorem 2.17) without truncation.

**Maximum Mean Discrepancy (MMD)** between two sets of paths P, Q:

```
D_sig(P, Q)² = ||ES(P) - ES(Q)|| = ⟨m(P) - m(Q), m(P) - m(Q)⟩_H
```

**Unbiased estimator**:
```
D_u(P, Q)² = (1/n(n-1)) Σ_{i≠j} k(x_i, x_j) 
           - (2/mn) Σ_i Σ_j k(x_i, y_j) 
           + (1/m(m-1)) Σ_{i≠j} k(y_i, y_j)
```

### 2.3 Path Transformations

Common preprocessing transforms applied before signature computation:

1. **Time normalization**: ϕ_time: normalizes time to [0,1]
2. **State normalization**: ϕ_norm: divides by initial value x_i/x_0
3. **Increment transform**: ϕ_incr: captures path volatility as first-order effect

---

## 3. Algorithm: Online Regime Detection

### 3.1 Pseudocode: Sub-path Extraction

```
INPUT: Time series x = (x_1, ..., x_N), hyperparameters h = (h1, h2)

1. Create time-augmented path: x̂ = {(t_i, x_i)} for i = 1, ..., N
2. Apply stream transformer Φ (e.g., time + state normalization)

3. Extract sub-paths (Definition 3.1):
   SP_h(x̂) = {s_j = (x̂_{j*h1}, ..., x̂_{(j+1)*h1 - 1})} for j = 0, ..., N1-1
   where N1 = floor(N / h1)

4. Build ensemble paths (Definition 3.2):
   EP_h(x̂) = {s_k = (s_j)_{j=k}^{k+h2-1}} for k = 0, ..., N2-1
   where N2 = N1 - h2
```

### 3.2 Pseudocode: Auto-Evaluator (Non-parametric Detection)

```
INPUT: Ensemble paths EP = (s_1, ..., s_M), lags L, confidence α

1. FOR i = max(L)+1 to M:
   a. Calculate L-lag auto evaluation score (eq. 29):
      A_L[i] = Σ_{l∈L} w_l * D_sig(s_{i-l}, s_i)
   
2. Build temporal prior distribution D_t from sliding window W:
   - Fit Gamma(α_t, β_t) to recent MMD scores
   - α_t = E[D_t]² / Var(D_t)
   - β_t = W * Var(D_t) / E[D_t]

3. Compute critical threshold c_α at (1-α) quantile of fitted distribution

4. Detect regime change if A_L[i] > c_α

OUTPUT: Score vector A_L, threshold c_α, regime change points
```

### 3.3 Pseudocode: MMD with Prior Beliefs

```
INPUT: Beliefs P = (P_1, ..., P_k), ensemble paths EP, confidence α

1. FOR each belief P_j:
   a. Simulate N sample paths from model P_j
   b. Bootstrap null distribution D_j of D_sig under P_j
   c. Compute critical value c_j^α as (1-α) quantile of D_j

2. FOR each ensemble s_i in EP:
   a. Calculate score vector (eq. 27):
      α(s_i) = [ (1/n) Σ_l D_sig(s_i, x_l^j) ] for j = 1, ..., k
      where x_l^j are samples from belief P_j

3. Regime classification:
   - If α(s_i)_j < c_j^α: s_i conforms to belief P_j
   - If α(s_i) fails all thresholds: s_i is anomalous

OUTPUT: Score matrix Λ, regime classifications
```

---

## 4. Mathematical Formulation

### 4.1 Signature Kernel PDE (Theorem 2.17)

The signature kernel k_sig(X, Y) = f(T, T) where f solves:

```
∂²f/∂s∂t = f(s,t) * ⟨dX_s, dY_t⟩_V
with f(0, t) = f(s, 0) = 1
```

This allows computation without explicit signature evaluation (kernel trick).

### 4.2 Rank-2 Signature MMD (Definition 2.20)

For non-Markovian processes, use rank-2 signature:

```
D²_sig(P, Q) = ||ES²(P) - ES²(Q)||_{T²(E)}
```

where T²(E) = T(T(E)) is the iterated tensor algebra.

**Theorem 2.21**: Rank-2 MMD metrizes the rank-1 adapted topology:
```
D²_sig(X, Y) = 0 ⟺ X ∼_1 Y (equality in conditional law)
```

---

## 5. Implementation Notes

### 5.1 Hyperparameter Selection

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| h1 | Sub-path length | 7-21 (days for daily data) |
| h2 | Ensemble size | 8-16 |
| σ (RBF scale) | Kernel smoothing | 0.01-1.0 (via bootstrap) |
| α | Confidence level | 0.05 (95% threshold) |
| W | Sliding window | 100-250 observations |

### 5.2 Computational Complexity

- Signature kernel evaluation: O(n * d) where n = path length, d = dimension
- MMD computation: O(h1² * h2²) for ensembles of size h2 with paths of length h1
- Rank-2 MMD: ~100x slower than Rank-1

### 5.3 Edge Cases

1. **Non-Markovian data**: Use rank-2 MMD or path-by-path similarity score Σ_{P,Q}
2. **Single-path evaluation**: Use signature kernel scoring rule (Definition 2.25)
3. **High dimensions**: Signature kernel is linear in d, scales well
4. **Missing data**: Linear interpolation for bounded variation paths

---

## 6. Key Results from Paper

### 6.1 Synthetic Data Experiments

| Method | Accuracy (Total) | ROC AUC | Runtime |
|--------|-----------------|---------|---------|
| D¹_sig (Rank-1 MMD) | 93.6% | 0.992 | 3.9s |
| D^N_sig (Truncated) | 72.7% | 0.875 | 5.1s |
| SIG-CON (Signature Conformance) | 82.2% | 0.835 | 45.0s |

### 6.2 Real Data Applications

- **US Equities (1980-2023)**: Successfully detected 1987 crash, dot-com bubble, 2008 GFC, COVID-19 volatility
- **Cryptocurrencies (2019-2022)**: Identified bull run periods and subsequent corrections
- **Correlation with VIX**: MMD scores track VIX well with appropriate lag

---

## 7. Dependencies for Implementation

```python
# Core
numpy, pandas, scipy

# Signature computation
signatory  # PyTorch-based, efficient
# OR
iisignature  # NumPy-based, simpler

# Kernel methods
sklearn (for clustering)
torch (if using signatory)

# Statistical
scipy.stats (Gamma fitting)
```

---

## 8. References

- [CO18] Chevyrev & Oberhauser - Signature kernel characterization
- [SCF+21] Salvi et al. - Kernel PDE for signature computation
- [SLL+21] Salvi et al. - Higher-rank signatures
- [HIM21] Horvath et al. - Previous work on Wasserstein k-means regime detection
- [BHL+20] Buhler et al. - Deep hedging, sig-MMD for market generation

---

## Evaluation

### Scores

| Metric | Score (out of max) | Notes |
|--------|-------------------|-------|
| Readme Quality | 8/10 | Excellent mathematical depth with proper PDE description of signature kernel, rank-2 MMD, path transformations. Pseudocode covers sub-path extraction, auto-evaluator, MMD with prior beliefs, and math formulation |
| Code Quality | 6/10 | Self-contained implementation but functionality is focused on synthetic data testing and ensemble-based MMD computation. No proper output generation functions. Lacks main() execution path that produces spec-compliant outputs |
| Output Compliance | 0/10 | **No CSV or JSON output files produced.** Output is a text file listing 25 detected change points from 2010-2020. No per-row regime predictions for the evaluation window (2021-2026). The output format is not machine-readable as required |
| Output Quality | 5/10 | Detected 25 regime transitions (2010-2020) and identifies key events (European Debt Crisis, COVID crash). However, the data only covers 3522 days (2010-2020), missing most of the required evaluation window (2021-2026) |
| Adherence to T2 Spec | 0/10 | Zero compliance — no CSV, no JSON, wrong data range, wrong output format |
| **Total** | **19/50** | |

### Strengths
- Strong mathematical understanding of the paper — only implementation to discuss the signature kernel PDE and rank-2 signatures
- Good detection of historical events (European Debt Crisis 2010, US Gov Shutdown 2018, COVID-19 2020)
- 25 regime changes detected is a reasonable number

### Weaknesses
- **Output data range (2010-2020) does not cover the required evaluation window (2021-2026) at all**
- No spec-compliant output files produced — cannot be evaluated programmatically
- Code lacks output generation pipeline
- Data source appears to be different from the shared sp500_ohlcv_5yr.parquet
