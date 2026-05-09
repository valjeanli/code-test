# Paper Implementation: SSRN 3947905 / arXiv 2306.15835

## Paper Details
- **Title**: Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures
- **Authors**: Blanka Horvath (Oxford/KCL), Zacharia Issa (KCL)
- **Year**: 2023
- **Venue**: arXiv (preprint)

## Problem Statement
Financial time series exhibit non-stationarity and volatility clustering. The paper addresses two problems:
1. **Market Regime Detection Problem (MRDP)**: Online (real-time) detection of distributional changes in financial paths
2. **Market Regime Classification Problem (MRCP)**: Grouping historical periods into clusters of similar market behavior

## Core Methodology

### 1. Path Signatures (Feature Map)
The **signature** of a path X: [a,b] -> R^d is the sequence of iterated integrals:

```
S(X)_{[a,b]} = (1, X^1_{[a,b]}, ..., X^N_{[a,b]}, ...)
```

where X^k = integral over a < t1 < ... < tk < b of dX_{u1}^{i1} (x) ... dX_{uk}^{ik}

The truncated signature at order N captures path information including:
- Level 1: increments (returns)
- Level 2: area under the path (captures autocorrelation)
- Level 3+: higher-order interactions

**Key advantage**: Captures order-dependent (path-dependent) information, unlike distributional approaches.

### 2. Signature Kernel
The signature kernel computes inner products in signature space without explicit truncation:

```
k_sig(x, y) = <S(x), S(y)>_T((V))
```

The **RBF-lifted signature kernel** applies an RBF kernel phi before computing signatures:
```
k_sig^phi(x, y) = <S(phi(x)), S(phi(y))>
```

This is universal (dense in continuous functions on compact sets).

### 3. Maximum Mean Discrepancy (MMD)
The unbiased MMD estimator between distributions P and Q:

```
D_u^k(P,Q)^2 = 1/(n(n-1)) sum_{i!=j} k(xi,xj) - 2/(mn) sum_i sum_j k(xi,yj) + 1/(m(m-1)) sum_{i!=j} k(yi,yj)
```

When using the signature kernel, this becomes a **path-wise two-sample test** that can detect distributional changes in the underlying path dynamics.

### 4. Online Detection Algorithm (Section 4)

**Sub-path decomposition** (Def 3.1):
Given hyperparameter h = (h1, h2), decompose path into overlapping sub-paths of length h1, grouped in ensembles of h2.

**L-lag Auto Evaluation Score** (Eq 29):
```
A_L(s_hat)_i = sum_{l in L} w_l * D_sig^r(s_{i-l}, s_i)
```

This compares the current ensemble s_i against lagged ensembles s_{i-l} using the signature MMD. Weights w_l can be uniform or exponentially decaying.

**Detection rule**: If A_L exceeds a threshold c_alpha (derived from an empirical prior of historical MMD scores at quantile alpha), declare anomalous / regime change.

### 5. Regime Clustering (Section 5)
Uses **sliced Wasserstein k-means** on signature features:
- Extract signature features from each sub-path
- Compute sliced Wasserstein distance between feature distributions
- Apply k-means clustering with Wasserstein barycenters as centroids

## Implementation Notes

### Practical Adaptation for S&P 500 Daily Data
The paper's experiments use high-frequency data. For daily S&P 500 data, we adapt:

1. **Feature extraction**: Compute rolling window features (returns, volatility, momentum) as proxies for signature features
2. **Signature approximation**: Use truncated path signatures (order 1-3) on rolling windows of log returns
3. **MMD computation**: Use Gaussian (RBF) kernel MMD between reference and test windows
4. **Online detection**: Compare MMD score against rolling empirical distribution of historical scores
5. **Regime assignment**: Post-hoc clustering of detected change-point segments using k-means

### Key Parameters (from paper Section 6.1)
- h = (h1, h2) = (8, 8): window length and ensemble size for daily data
- alpha = 0.95: detection threshold quantile
- Empirical prior: last 200 MMD scores
- Signature truncation order: N = 4 (for computational feasibility)
- RBF kernel sigma = 1.0

### Algorithm Pseudocode

```
INPUT: price path P[0..T], hyperparams h=(h1,h2), alpha, prior_window
OUTPUT: regime labels for each trading day

1. Compute log returns: r[t] = log(P[t]/P[t-1])
2. For each window starting at t:
   a. Extract sub-path s[t] = r[t:t+h1]  (returns in window)
   b. Compute truncated signature Sig(s[t]) up to order N
3. For each ensemble index i from max(L) to N2:
   a. Form ensemble S_i = {s[i-h2+1], ..., s[i]}  (h2 consecutive sub-paths)
   b. For each lag l in L:
      - Form lagged ensemble S_{i-l}
      - Compute MMD(S_i, S_{i-l}) using signature kernel
   c. A_L[i] = sum_l w_l * MMD(S_i, S_{i-l})
4. For each time point:
   a. Compare A_L[i] to threshold = quantile(prior_scores, alpha)
   b. If A_L[i] > threshold: flag as regime change
5. Post-process: assign regime labels to segments between change points
6. Cluster regimes using k-means on segment feature vectors
```

### Time/Space Complexity
- Signature computation: O(d * h1^N) per window where d = dimension, N = truncation order
- MMD computation: O(h2^2) per ensemble pair
- Total: O(T * h2^2 * |L|) for the online scan
- Space: O(prior_window) for the empirical prior

### Deviations from Paper
1. We use a simplified signature computation (truncated, not kernel-trick based) for computational feasibility
2. For univariate S&P 500 data, the path is 1-dimensional (log returns), so we augment with time
3. We use scikit-learn k-means for clustering instead of the Wasserstein k-means from the paper
4. The paper uses the "auto evaluator" for fully non-parametric detection; we implement a practical version with RBF kernel MMD

## References
- Horvath, Issa (2023). arXiv:2306.15835
- Chevyrev, Oberhauser (2022). Signature moments to characterize laws of stochastic processes
- Liu et al. (2020). Learning with Wasserstein features via RKHS embeddings

---

## Evaluation

### Scores

| Metric | Score (out of max) | Notes |
|--------|-------------------|-------|
| Readme Quality | 7/10 | Good coverage of paper methodology, clear pseudocode with auto-evaluator and ensemble extraction, implementation notes, complexity analysis. Less detailed than top readmes but solid |
| Code Quality | 7/10 | 26KB codebase implementing proper truncated signatures, auto-evaluator with weighted L-lag detection, k-means clustering. Produces two output files |
| Output Compliance | 5/10 | Has TWO output CSV+JSON sets. Main CSV (`sig_mmd_regimes.csv`) has **predicted_regime=-1** for ALL rows, violating the spec (must be 0 <= regime < num_regimes). The `sig_mmd_online_regimes.csv` is well-formed with proper columns, but metadata label map conflicts (shows 0=bear, 1=bull, 2=sideways but data shows 0=bull). Confidence is flat 0.5 |
| Output Quality | 5/10 | The online detector output covers 2006-2025 with 5032 rows. However, regime 0 (labeled "bull") is assigned to ALL rows — the algorithm detected no regime changes at all in the full 20-year period. The main detector (`sig_mmd_regimes.csv`) has -1 for all rows, effectively failing |
| Adherence to T2 Spec | 5/10 | Produces the right file structure but with significant data issues (-1 values, label conflicts). The `sig_mmd_online_regimes.csv` is closer to compliance but has zero transitions detected |
| **Total** | **29/50** | |

### Strengths
- Two different detection approaches (batch and online)
- Proper truncated signature computation (not just rolling stats)
- Good auto-evaluator implementation with weighted multi-lag scoring
- CSV has correct columns and format

### Weaknesses
- **Main output file has -1 regime values — violates spec constraint `0 <= predicted_regime < num_regimes`**
- **Online detector detects zero regime changes in 20 years** — all rows have regime 0 with flat 0.5 confidence
- Confusing dual output files — unclear which is the canonical output
- Metadata label map doesn't match actual data labels
