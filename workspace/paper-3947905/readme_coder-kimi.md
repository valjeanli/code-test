# Paper Readme: SSRN 3947905 / arXiv 2306.15835

## Paper Metadata
| Field | Value |
|-------|-------|
| **Title** | Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures |
| **Authors** | Zacharia Issa, Blanka Horvath |
| **Year** | 2023 |
| **arXiv ID** | 2306.15835 |
| **SSRN ID** | 3947905 |
| **Categories** | stat.ML, q-fin.MF |
| **PDF** | https://arxiv.org/pdf/2306.15835v1 |
| **Code** | https://github.com/issaz/signature-regime-detection/ |

---

## Problem Statement

Financial time series are **non-stationary** and exhibit **volatility clustering**.  Traditional parametric models (e.g. GARCH, Markov-switching) require strong distributional assumptions and often fail to capture path-dependent effects such as autocorrelation or non-Markovian dynamics.

The paper addresses two complementary problems:

1. **Market Regime Detection Problem (MRDP)** — online detection of regime changes in streaming path data.
2. **Market Regime Clustering Problem (MRCP)** — offline identification of distinct periods of similar market activity within a single long path.

Both problems are cast on **path space** (the space of continuous functions) rather than on the real line, which allows the methodology to capture path-dependent features like lead-lag effects and serial dependence.

---

## Core Methodology — High Level

The methodology rests on three pillars from rough-path theory:

1. **Path Signature** (`S(X)`) — a universal feature map that converts a path into an infinite sequence of iterated integrals.  By the universality theorem, linear functionals of the signature approximate any continuous functional on path space.
2. **Signature Kernel** (`k_sig(x,y) = <S(x), S(y)>`) — the inner product in the signature RKHS.  The paper uses both the *truncated* signature kernel (exact finite sum up to depth M) and the *full* signature kernel computed via a Goursat PDE.
3. **Maximum Mean Discrepancy (MMD)** — a distance between probability distributions in the signature RKHS, used as a non-parametric two-sample test.

The key insight is that **distributions on path space** (ensembles of short path segments) can be compared via the MMD.  When two consecutive ensembles are drawn from different distributions, the MMD score jumps, signaling a regime change.

---

## Mathematical Formulation

### Path Signature (Definition 2.1)
For a path `X: [0,T] -> R^d` of bounded variation, the **signature** of order `k` is the tensor

```
S^k(X) = int_{0<t1<...<tk<T}  dX_{t1} \otimes ... \otimes dX_{tk}
```

The full signature is the infinite sequence `S(X) = (1, S^1(X), S^2(X), ...)`.

In practice, for a discrete piecewise-linear path `x = (x_0, x_1, ..., x_L)` with increments `dx_i = x_i - x_{i-1}`, the signature is computed recursively via the Chen identity (implemented in `iisignature`).

### Expected Signature (Definition 2.2)
For a measure `mu` on path space, the **expected signature** is

```
E_mu[S(X)] = int S(x) d mu(x)
```

The expected signature uniquely characterises the law of the process (characteristicness).

### Signature Kernel (Definition 3.3)
For two paths `x, y`, the **signature kernel** is

```
k_sig(x, y) = <S(x), S(y)>_T
```

where `<,>_T` is the inner product on the tensor algebra with factorial weighting.  The paper derives a PDE formulation (Goursat problem) allowing exact computation *without truncation*:

```
\partial_u \partial_v K(u,v) = <\dot{x}_u, \dot{y}_v> * K(u,v)
```

with boundary conditions `K(0,·)=K(·,0)=1`.

### Maximum Mean Discrepancy (Definition 3.1)
Given a characteristic kernel `k` and two measures `P, Q`:

```
MMD_k(P,Q) = || E_P[\phi(X)] - E_Q[\phi(Y)] ||_H
```

where `\phi` is the feature map into the RKHS `H` associated with `k`.  Equivalent closed form:

```
MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2 E[k(X,Y)]
```

### Unbiased Estimator (Eq. 11–13)
For finite samples `X = {x_1,...,x_m}`, `Y = {y_1,...,y_n}`:

```
MMD^2_u =  1/(m(m-1)) * sum_{i \neq j} k(x_i, x_j)
         + 1/(n(n-1)) * sum_{i \neq j} k(y_i, y_j)
         - 2/(m n)     * sum_{i,j}     k(x_i, y_j)
```

This is the U-statistic used in the paper's experiments.

### MMD Two-Sample Test (Theorem 3.2)
Under `H0: P = Q`, as `m,n -> inf`, the empirical MMD converges to the population MMD at rate `O(1/sqrt(m+n))`.  The test rejects `H0` when `MMD > c_{alpha}` where `c_{alpha}` is the `(1-alpha)` quantile of the null distribution.

### Higher-Rank Signatures (Definition 3.5–3.7)
The paper also introduces **rank-2** (and higher) signatures that incorporate filtration information.  The rank-2 signature captures conditional distributions, making it strictly better for non-Markovian processes at a significant computational cost (~100x slower).

---

## Core Algorithm Pseudocode

### ALGORITHM: MMD-DET (Online Regime Detector)
**Input:**  Stream of paths `P_1, P_2, ...`; confidence level `alpha`; signature depth `M`; ensemble sizes `B` (reference) and `b_test` (test).
**Output:** Set of detected change-point indices `C`.

```
C <- empty set
i <- 0                                    # current position in stream

while i + B + b_test <= len(stream):
    R <- {P_i, ..., P_{i+B-1}}            # reference ensemble
    T <- {P_{i+B}, ..., P_{i+B+b_test-1}} # test ensemble

    # Compute signature features
    Phi_R <- [S_M(p) for p in R]         # truncated signature, depth M
    Phi_T <- [S_M(p) for p in T]

    # Unbiased MMD estimate
    mmd2 <- MMD^2_u(Phi_R, Phi_T)

    # Bootstrap critical value
    c_alpha <- BootstrapThreshold(Phi_R, Phi_T, alpha)

    if mmd2 > c_alpha:
        C <- C union {i+B+b_test-1}       # change point at end of test window
        i <- i + B + b_test                # jump ahead, start new reference
    else:
        i <- i + b_test                    # slide forward (or step=1)

return C
```

### ALGORITHM: Bootstrap Threshold
**Input:**  Feature matrices `X` (m x d), `Y` (n x d); bootstrap replications `B`; quantile `alpha`.
**Output:** Critical value `c_alpha`.

```
Pool <- concatenate(X, Y)                  # (m+n) x d
stats <- empty list
for b = 1 to B:
    perm <- random permutation of Pool
    X_b <- perm[0:m]
    Y_b <- perm[m:m+n]
    stats.append( MMD^2_u(X_b, Y_b) )
c_alpha <- quantile(stats, 1-alpha)
return c_alpha
```

### ALGORITHM: Agglomerative Clustering (MRCP)
**Input:**  Set of paths `{P_1,...,P_N}`; desired clusters `K`; signature depth `M`.
**Output:** Cluster labels `l_1,...,l_N`.

```
# Compute pairwise MMD distance matrix
for i = 1 to N:
    for j = i+1 to N:
        D[i,j] <- D[j,i] <- sqrt( max(0, MMD^2_u(S_M(P_i), S_M(P_j))) )

# Hierarchical clustering
labels <- AgglomerativeClustering(D, n_clusters=K, linkage='average')
return labels
```

---

## Key Assumptions & Constraints

| # | Assumption | Practical Impact |
|---|-----------|------------------|
| 1 | Paths are **piecewise linear interpolants** of discrete data. | The signature is well-defined because discretised paths are of bounded variation.  The finer the sampling, the closer to continuous-time limit. |
| 2 | **Time reparametrisation invariance** of the signature. | One can normalise the time axis to `[0,1]` without losing information about path shape. |
| 3 | MMD kernel is **characteristic / universal**. | The signature kernel satisfies this (Proposition 3.4), so MMD=0 iff distributions are equal in law. |
| 4 | Samples are **i.i.d. within each regime**. | Path ensembles are constructed by sliding a non-overlapping (or slightly overlapping) window.  The paper acknowledges that samples are not strictly independent but the U-statistic is robust. |
| 5 | The **response time** of the detector depends on `b_test`. | Smaller `b_test` gives faster reaction but higher variance and more Type-II errors.  The paper uses `b_test` in the range 5–20 paths. |
| 6 | **Higher-rank signatures** require significantly more compute. | Rank-2 MMD is ~100x slower than rank-1.  For online settings the paper defaults to rank-1 unless non-Markovianity is suspected. |

---

## Implementation Notes (coder-kimi)

### What was implemented
- **Truncated signature map** (`iisignature.sig`, depth=4).  This corresponds to "MMD-T" in the paper.
- **Unbiased MMD^2 estimator** using matrix-vector products for efficiency.
- **Bootstrap threshold** with 300 replications at 95% confidence.
- **Online sliding-window detector** with configurable reference/test ensemble sizes.
- **Agglomerative hierarchical clustering** via `sklearn` with MMD precomputed distance matrix.
- **Feature normalisation** (z-score) before MMD computation — not explicitly in the paper but improves numerical stability when signature dimension varies.

### Why not the full signature kernel (Goursat PDE)?
The paper's preferred method is the *untruncated* signature kernel computed by solving a hyperbolic PDE.  However:
1. The PDE solver is not available as a standard pip-installable package.
2. The truncated method (MMD-T) is explicitly benchmarked in the paper and shown to work well for modest truncation depths (4–5).
3. For S&P 500 daily data, depth-4 captures sufficient information (up to 4th-order iterated integrals).

### Path construction details
- Raw data: S&P 500 daily close prices.
- Transform 1: log-prices → log-returns (`diff(log(close))`).
- Transform 2: time normalisation is implicit because we treat each path as a sequence of `L` equally spaced increments.
- Each path is a `(L, 1)` numpy array fed to `iisignature.sig`.

### Parameters chosen
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `path_length` (`L`) | 21 | One trading month (~21 business days). |
| `n_reference` (`B`) | 10 | ~10 months of reference history. |
| `n_test` (`b_test`) | 10 | Same as reference for symmetry; slide step = 10 for speed. |
| `sig_depth` (`M`) | 4 | Balances expressiveness vs. dimensionality.  Signature dim for d=1, M=4 is 5. |
| `bootstrap_alpha` | 0.95 | Standard confidence used in paper experiments. |
| `n_bootstrap` | 300 | Stable empirical quantile. |

### Computational complexity
- Signature computation per path: `O(L * d^M)` — dominated by `iisignature` C++ code.
- MMD between two ensembles: `O((B+b_test)^2 * sig_dim)`.
- Per iteration (one test/reference pair): ~a few milliseconds for our parameters.
- Total runtime on 14k daily points: < 30 seconds on CPU.

---

## Input / Output Specification

### Input
- **Data file**: CSV with columns `Date` (YYYY-MM-DD) and `Close` (float).
- **Source**: S&P 500 daily close from Yahoo Finance (`^GSPC`), 1970–present.

### Output Files
| File | Description |
|------|-------------|
| `output_coder-kimi.txt` | Detected change points with dates, full MMD trajectory, configuration settings, and clustering summary. |
| `regime_detection.png` | Visualisation: S&P 500 price series overlaid with MMD^2 scores, bootstrap thresholds, and vertical lines marking regime changes. |

### Output format (text)
```
Date: YYYY-MM-DD  |  MMD^2: <float>
```
One line per detected change point, followed by the first 30 rows of the full MMD score table.

---

## Edge Cases & Deviations

1. **Bootstrap variance on small ensembles**: if `B` or `b_test` is small (<5), the bootstrap null distribution is unstable.  The paper recommends `B >= 5`; we use `B=10`.
2. **Type-II error during gradual regime changes**: the MMD detector may miss slow drifts and only fire once the distribution has shifted sufficiently.  This is acknowledged in the paper (Section 4.2).
3. **Choice of path transform**: the paper also discusses *lead-lag* transforms for capturing temporal ordering in higher dimensions.  We stick to the basic increment transform because the S&P 500 is univariate.
4. **Rank-2 limitation**: the rank-2 signature MMD is *not* implemented in this code because it requires a conditional embedding and custom PDE solvers.  The paper notes it is ~100x slower; for online detection with sequential data, rank-1 is the practical default.

---

## References (from paper)
- Chevyrev & Kormilitzin (2016) — signature moments characterise laws of stochastic processes.
- Kiraly & Oberhauser (2019) — kernels for sequentially ordered data.
- Salvi et al. (2021) — the signature kernel is the solution of a Goursat PDE.
- Gretton et al. (2012) — a kernel two-sample test.
- Chevyrev & Lyons (2016) — signature uniqueness for bounded variation paths.
- Fermanian (2021) — clustering market regimes using the Wasserstein distance (prior work by same authors).

---

## Evaluation

### Scores

| Metric | Score (out of max) | Notes |
|--------|-------------------|-------|
| Readme Quality | 10/10 | Most detailed readme — 265 lines covering paper metadata, mathematical formulation (signature, expected signature, signature kernel, MMD, unbiased estimator, two-sample test, higher-rank signatures), two complete algorithm pseudocodes (MMD-DET and BootstrapThreshold), edge cases table, and comprehensive implementation notes |
| Code Quality | 7/10 | Uses `iisignature` library for proper truncated signature computation (depth 4). Implements unbiased MMD^2 estimator with bootstrap threshold (300 replications). Well-structured but lacks output generation for spec compliance |
| Output Compliance | 0/10 | **No CSV or JSON output files.** Output is a text file with MMD scores and 5 detected change points spanning 1970-2026. No per-row regime predictions. The output format does not match the required spec at all |
| Output Quality | 6/10 | Successfully detected 5 major market events (1987 crash, 1998 LTCM, 2008 GFC, 2020 COVID). With 14,209 data points covering 1970-2026. Uses proper signature computation. However, only 5 change points in 50+ years is too conservative (98th percentile global threshold) |
| Adherence to T2 Spec | 0/10 | Zero compliance — no required CSV columns, no metadata JSON, wrong output format |
| **Total** | **23/50** | |

### Strengths
- **Only implementation using proper path signature library** (`iisignature`) for actual signature computation
- Most detailed paper readme — shows genuine understanding of the mathematical foundations
- Bootstrap threshold with 300 replications is statistically rigorous
- Detected historically correct events (1987 crash, 2008 GFC, 2020 COVID)
- Good edge case awareness documented

### Weaknesses
- **No spec-compliant output files at all** — cannot be programmatically evaluated against the scoring rubric
- Only 5 change points in 50+ years is too conservative for practical use
- Uses a global 98th percentile threshold rather than the paper's adaptive threshold
- No regime label assignment — just change point detection
