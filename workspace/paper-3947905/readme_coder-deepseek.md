# Regime Detection via Signature MMD — Paper SSRN 3947905 / arXiv 2306.15835

## Paper Details

- **Title**: Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures
- **Authors**: Blanka Horvath (Oxford, Oxford-Man Institute, Alan Turing Institute) & Zacharia Issa (King's College London)
- **Published**: June 27, 2023 (arXiv:2306.15835, SSRN 3947905)
- **Venue**: Working paper

## Problem Statement

Financial time series exhibit non-stationarity, volatility clustering, and heteroscedasticity. The paper addresses two related problems:

1. **Market Regime Detection Problem (MRDP)**: Online (real-time) detection of shifts in the underlying distribution of asset returns — "is a regime change happening now?"
2. **Market Regime Clustering Problem (MRCP)**: Offline grouping of historical periods into regimes with similar distributional characteristics.

The key innovation: operating on **path space** (distributions over entire price trajectories) rather than on individual return distributions, using rough path signatures as features and signature-kernel MMD as the distance metric.

## Core Methodology

### 1. Data Preprocessing (Section 3.1)

Given a time-augmented path ŝ (e.g., daily S&P 500 prices with timestamps):

```
ŝ = {(t_i, s_i) : s_i ∈ ℝ^d, i=1,...,N}
```

**Step 1: Path Transformation** Φ = φ_time ∘ φ_norm ∘ φ_scale^λ

- φ_scale^λ: Scale by λ = dt^(-1/2) where dt = 1/252 (annualization factor for daily data)
- φ_norm: Normalize via cumulative sum (lead-lag embedding to convert returns to a path)
- φ_time: Add time augmentation (append time coordinate)

**Step 2: Sub-path Extraction** (Definition 3.1)
- Hyperparameters h = (h1, h2):
  - h1 = sub-path length (e.g., 21 days = ~1 month of trading)
  - h2 = ensemble size (e.g., 12 consecutive sub-paths)
- Extract sequential sub-paths 𝒮𝒫_h(ŝ) = (s_0, s_1, ..., s_{N_1})
- Form ensembles ℰ𝒫_h(ŝ) = (e_0, e_1, ..., e_{N_2}) where each e_i contains h2 consecutive sub-paths

### 2. Signature-Based Feature Map (Section 2.1-2.3)

**Path Signature** (Definition 2.1):
- The signature S(X) of a path X: [a,b] → ℝ^d is an infinite series of tensors:
  S(X) = (1, 𝕏¹, 𝕏², 𝕏³, ...) where 𝕏^k ∈ (ℝ^d)^{⊗k}
- Iterated integrals capture all path information — S(X) uniquely characterizes the path up to tree-like equivalence

**Signature Kernel** (Equation 17):
- k_sig(x, y) = ⟨S(x), S(y)⟩_{T((E))} — inner product in the tensor algebra
- Can be computed efficiently via a kernel trick (PDE-based) without truncation
- RBF-lifted variant: k_sig^φ = k_sig ∘ φ where φ is an RBF feature map → makes the kernel universal on compact path sets

### 3. Maximum Mean Discrepancy (Equation 16)

The MMD between two distributions ℙ and ℚ using kernel κ is:

```
𝒟_u^κ(ℙ, ℚ)² = 1/(n(n-1)) Σ_{i≠j} κ(x_i, x_j)           [within-X similarity]
               - 2/(mn) Σ_i Σ_j κ(x_i, y_j)               [cross similarity]
               + 1/(m(m-1)) Σ_{i≠j} κ(y_i, y_j)           [within-Y similarity]
```

This is an unbiased estimator of the squared MMD. Larger values indicate the samples are more likely from different distributions.

### 4. Online Regime Detection: Auto-Evaluator (Definition 3.9)

**Parametric Mode** (Section 3.2.1) — modeller holds prior beliefs 𝔓:
- Pre-simulate paths from each regime model ℙ_θ_i
- Compute critical values c_α under H0 (no regime change) via bootstrap
- For each new ensemble, compute MMD against each belief → if > c_α at (1-α)% confidence, flag as regime change

**Non-Parametric Mode: Auto-Evaluator** (Section 3.2.2) — no prior beliefs:
- Compare each ensemble e_i against lagged ensembles e_{i-l} for l ∈ L
- Score vector: A_L(ŝ)_i = Σ_{l∈L} w_l · 𝒟_sig^r(s^{i-l}, s^i)  (Equation 29)
- L = {4, 8, 12} for multi-scale detection
- Build empirical prior from previous 200 MMD scores
- Critical threshold c̄_t^α at (1-α)% confidence level
- Ensembles scoring above threshold are anomalous → regime change detected

### 5. Regime Clustering (Section 3.3)

For the MRCP:
- Compute signature kernel Gram matrix K_{ij} = k_sig(s_i, s_j) for all sub-paths
- Apply agglomerative hierarchical clustering on the induced distance matrix
- Results compared favorably against Wasserstein k-means baseline

### Hyperparameters used in the paper

| Parameter | Real Data Value | Description |
|-----------|----------------|-------------|
| h = (h1, h2) | (8, 8) — daily; (21, 12) — monthly | Sub-path length, ensemble size |
| λ | dt^(-1/2) = sqrt(252) | Annualization scaling |
| σ | 1.0 | RBF kernel bandwidth |
| α | 0.95 | Confidence level |
| Prior window | 200 scores | Empirical null distribution size |
| L | {4, 8, 12} | Multi-scale lag set |
| Detector rank | r = 1 (fast) or r = 2 (accurate) | MMD variant |
| Signature truncation | None (kernel trick) | Infinite-dimensional feature map |

## Algorithm Pseudocode

```
ALGORITHM: SigMMD Regime Detection (Auto-Evaluator)
INPUT: Price series p[0..T], hyperparameters h=(h1,h2), L, α, σ
OUTPUT: Regime labels and transition dates

1. PREPROCESS:
   Compute log returns r[t] = log(p[t] / p[t-1])
   Apply scaling: r_scaled[t] = r[t] * sqrt(252)  // annualize
   Time-augment: add time channel
   
2. EXTRACT SUB-PATHS:
   Sub-paths s_j of length h1 (sliding window, stride = 1)
   N1 = T - h1 + 1 sub-paths
   
3. FORM ENSEMBLES:
   Ensemble e_i = {s_i, s_{i+1}, ..., s_{i+h2-1}}
   N2 = N1 - h2 + 1 ensembles
   
4. COMPUTE SIGNATURES:
   For each sub-path s_j:
     Compute truncated signature Sig(s_j) up to level M (M=3 for 2D paths)
   Store as feature vectors
   
5. BUILD REFERENCE PRIOR (first 200 ensembles):
   For i = 0 to min(200, N2):
     Compute MMD scores using auto-evaluator
   Build empirical null distribution D̄
   
6. ONLINE DETECTION (for each ensemble e_i, i > prior_window):
   A_i = Σ_{l∈L} w_l · MMD(e_{i-l}, e_i)  // Eq 29
   Critical value c = percentile(D̄, α)
   
   IF A_i > c:
     Mark ensemble i as regime change
   ELSE:
     Add A_i to null distribution D̄ (rolling window of 200)
   
7. AGGLOMERATIVE CLUSTERING:
   Compute pairwise MMD between all sub-paths
   Build distance matrix D_{ij} = MMD(s_i, s_j)
   Apply hierarchical clustering with Ward linkage
   
8. LABEL REGIMES:
   For each detected regime period:
     Compute mean return, volatility
     Label: "bull" (μ > 0, σ < median), "bear" (μ < 0, σ > median), 
            "high_vol" (σ > 75th pct), "low_vol" (σ < 25th pct)
   
RETURN: Date → regime_label mapping with transition flags
```

## Implementation Notes

1. **Truncated Signatures**: The full signature kernel requires the `sigkernel` package (PDE-based computation). Our implementation uses truncated signatures (level 2-3) as a practical approximation, which captures path shape (level 1), area/quadratic variation (level 2), and higher-order interactions (level 3).

2. **Path Dimension**: For a univariate price series with time augmentation, d = 2 (price channel + time channel).

3. **MMD Computation**: The inner product of signatures gives us the signature kernel value. For truncated signatures, we compute the explicit feature vectors and use linear kernel.

4. **Auto-Evaluator Lag**: L = {4, 8, 12} means we compare against 1, 2, and 3 months ago (for h1=21 trading days). This multi-scale approach captures both short-term and medium-term regime shifts.

5. **Real Data Results** (from paper, Section 6.1):
   - Successfully detected: 1987 crash, dot-com bubble (2000-2001), GFC (2008), European debt crisis, COVID-19 (2020)
   - MMD scores track VIX index well
   - Works on high-dimensional baskets (8 equities simultaneously)

## References

- [CO18] Chevyrev & Oberhauser (2018) — Signature MMD
- [SCF+21] Salvi, Cass, Foster, Lyons, Yang (2021) — Signature kernel trick
- [SLL+21] Salvi, Lemercier, Liu, Horvath, Lyons (2021) — Higher-rank signatures
- [HIM21] Horvath, Issa, Muguruza (2021) — Wasserstein k-means for regimes (predecessor paper)
- [LSD+20] Lyons et al. (2020) — Signature kernel universality

---

## Evaluation

### Scores

| Metric | Score (out of max) | Notes |
|--------|-------------------|-------|
| Readme Quality | 9/10 | Most thorough readme of all 6 — covers paper details, transformation steps (φ_scale, φ_norm, φ_time), auto-evaluator algorithm, agglomerative clustering, hyperparameter table, comprehensive pseudocode |
| Code Quality | 9/10 | 891 lines, well-organized with proper config dict, clear function separation, proper output generation per output_spec.json v1.0 (Section 7: generate_output_files), best docstrings and formatting |
| Output Compliance | 0/10 | **CRASHED during clustering (step 6).** Output file is only 28 lines showing "6. Clustering regimes via agglomerative clustering..." with no further output. No CSV or JSON files produced despite having proper output generation code |
| Output Quality | 0/10 | The algorithm detected 3896 regime changes out of 4988 valid scores — this is an astronomical 78% detection rate, which is clearly wrong (likely threshold too low or score normalization issue). No usable output data |
| Adherence to T2 Spec | 2/10 | Good intent — output generation functions are well-designed and strictly follow output_spec.json format. But no actual files were generated |
| **Total** | **20/50** | |

### Strengths
- Best code quality of all 6 — well-structured, modular, documented
- Output generation functions are the most spec-compliant design
- Best readme — excellent paper understanding
- Uses correct hyperparameters from paper (h1=21, h2=12, L=[4,8,12])
- Code has fallback mechanisms for NaN handling

### Weaknesses
- **Execution crashed — no usable outputs produced**
- **78% detection rate is clearly pathological** — likely a bug in score normalization or threshold calculation
- The output generation relies on ensemble-level regimes being reasonable, but the 78% detection rate suggests the raw detection is broken
- Uses truncated signatures (level 2) rather than full kernel trick, missing higher-order path information