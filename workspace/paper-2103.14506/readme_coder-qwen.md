# Asset Selection via Correlation Blockmodel Clustering

**Paper**: arXiv:2103.14506v2 [q-fin.PM]  
**Authors**: Wenpin Tang, Xiao Xu, Xun Yu Zhou (Columbia University, IEOR)  
**Date**: August 16, 2021 (v2; originally March 26, 2021)  
**Venue**: arXiv preprint  
**Pages**: 46 pages, 9 figures, 8 tables

---

## Problem Statement

Modern portfolio theory (Markowitz, 1952) suggests diversification across many assets, but including all available stocks (e.g., S&P 500's 500 stocks) is impractical for small investors/fund managers and increases overfitting risk. The key question:

> **How can we select a much smaller subset of the whole universe of stocks that achieves a sufficient level of diversification?**

The paper proposes a **two-stage decomposition**: first select a small, well-diversified subset of stocks, then apply Markowitz mean-variance optimization on this subset.

---

## Two Core Clustering Criteria

1. **Criterion 1**: Financial assets in the same group have **high correlations** (assets that move together should be grouped).
2. **Criterion 2**: Financial assets in the same group have **similar correlations with all other assets** (assets within a cluster are interchangeable in terms of their relationship with the rest of the universe).

Criterion 2 is the key innovation — it ensures that picking any representative from a cluster yields the same diversification properties.

---

## Correlation Blockmodel

### Model Setup (Eq. 1)

The standardized return of asset $i$ is decomposed as:

$$X^*_i = F_{z(i)} + U_i, \quad i \in [d]$$

where:
- $F = (F_1, \dots, F_K)$ are $K$ latent factors, one per cluster, with $E(F_k) = 0$
- $U = (U_1, \dots, U_d)$ are idiosyncratic fluctuations with $E(U_i) = 0$, $\text{Cov}(U_i, U_j) = 0$ for $i \neq j$, and $\text{Cov}(F_k, U_i) = 0$
- $z(i)$ maps asset $i$ to its cluster index

This implies: for $i, j$ in the same cluster and $l \neq i, j$: $\text{Corr}(X_i, X_l) = \text{Corr}(X_j, X_l)$

### Correlation Matrix Structure (Eq. 2)

$$\rho = Z \Sigma_F Z^\top + \Sigma_U$$

where $Z$ is the $d \times K$ membership matrix, $\Sigma_F = E(FF^\top)$, $\Sigma_U = E(UU^\top)$ (diagonal).

### Identifiability (Theorem 1, Eq. 3)

The unique coarsest partition $G^\star$ is defined by:

$$i \overset{G^\star}{\sim} j \iff \max_{l \neq i,j} |\rho_{il} - \rho_{jl}| = 0$$

---

## CORD Dissimilarity Measure (Eq. 4, 7)

**Population**: $\text{CORD}(i, j) := \max_{l \neq i,j} |\rho_{il} - \rho_{jl}|$

**Sample**: $\widehat{\text{CORD}}(i, j) := \max_{l \neq i,j} |\hat{\rho}_{il} - \hat{\rho}_{jl}|$

where $\hat{\rho}$ is the sample correlation matrix:

$$\hat{\rho} = \frac{1}{n-1} (X^\star)^\top X^\star \quad \text{(Eq. 6)}$$

Two assets are in the same cluster if their CORD distance is near zero — meaning they correlate with all other assets in the same way.

---

## PARTITION Procedure (Procedure 1)

Given a dissimilarity matrix $D$ and threshold $\varepsilon > 0$:

```
procedure PARTITION(D, ε):
    S ← [d]  (all assets), l ← 0
    while S ≠ ∅:
        l ← l + 1
        if |S| = 1:
            Ĝ_l ← S
        else:
            (i_l, j_l) ← argmin_{i,j ∈ S, i≠j} D(i, j)
            if D(i_l, j_l) > ε:
                Ĝ_l ← {i_l}  (singleton cluster)
            else:
                Ĝ_l ← {k ∈ S : min(D(i_l, k), D(j_l, k)) ≤ ε}
        S ← S \ Ĝ_l
    return Ĝ = {Ĝ_1, Ĝ_2, ...}
```

**Key insight**: The algorithm finds the most similar pair in the remaining set. If their dissimilarity exceeds $\varepsilon$, one becomes a singleton. Otherwise, all assets close to either core member join the cluster.

---

## ACC Algorithm — Full Pipeline (Algorithm 1)

```
procedure ACC(X, a, b, ng, U, k):
    Input: returns X (n×d), search range [a,b], grids ng,
           cluster range U, tail obs count k
    
    # Step 1: Standardize returns
    X* ← (X - mean(X)) / std(X)  (column-wise)
    
    # Step 2: Compute sample correlation matrix
    ρ̂ ← (1/(n-1)) (X*)ᵀ X*
    
    # Step 3: Compute CORD matrix
    CORD̂(i,j) ← max_{l≠i,j} |ρ̂_il - ρ̂_jl|
    
    # Step 4: Estimate heavy-tailedness α and constant L
    for each asset i:
        Y_i ← |(ρ̂^{-1/2} X*)_i|  (absolute whitened returns)
        Sort Y_i: Y_i[1] ≤ ... ≤ Y_i[n]
        Linear regression: log(Y_i[n-k:n-1]) ~ log(log(2n/[1:k]))
        α_i ← 1/slope, L_i ← exp(intercept)
    α ← min_i α_i, L ← max_i L_i
    
    # Step 5: Determine threshold search range (Rule 1)
    if n > (log d)^{4/α - 1}:
        T ← [a, b] × L² × sqrt(log d / n)
    else:
        T ← [a, b] × L² × (log d)^{2/α} / n
    
    # Step 6: Grid search over ε (Rules 2 & 3)
    Divide T into ng grids
    For each ε in grid:
        Ĝ_ε ← PARTITION(CORD̂, ε)
        if |Ĝ_ε| ∈ U:  (number of clusters in range)
            ρ̂_ave_ε ← avg intra-cluster correlation
        else:
            ρ̂_ave_ε ← -∞
    
    ε_T ← argmax_ε ρ̂_ave_ε
    return PARTITION(CORD̂, ε_T)
```

---

## Key Parameters (from paper's empirical setup)

| Parameter | Value | Description |
|-----------|-------|-------------|
| $n$ | 500 | Lookback trading days |
| $a, b$ | 0.1, 10 | Search range multipliers |
| $n_g$ | 100 | Number of grid points |
| $U$ | [15, 25] | Allowed number of clusters |
| $k$ | $n/4 = 125$ | Number of tail observations for $\alpha$ estimation |
| $\varepsilon$ upper cap | 2 | CORD max value |

---

## Theoretical Guarantees

### Theorem 2 — Optimal Asset Selection
Among all portfolios formed by picking one asset per cluster, the minimum-variance portfolio uses the **lowest-variance asset from each cluster**:
$$J^\star(k) = \arg\min_{j \in G_k} \text{Var}(X_j)$$

### Theorem 3 — Statistical Guarantee
Under $\alpha$-sub-exponential returns, if $\varepsilon$ satisfies:
$$\varepsilon \geq 2L^2 \left(c_1 \sqrt{\frac{\log d}{n}} + c_2 \frac{(\log d)^{2/\alpha}}{n}\right)$$
then PARTITION recovers the true partition $G^\star$ with probability $1 - 4/d$.

### Theorem 4 — Complexity
ACC runs in $O(nd^2 + d^3)$ arithmetic operations.

---

## Portfolio Construction

1. **Asset Selection**: Pick the lowest-volatility stock from each ACC cluster (Theorem 2)
2. **Allocation Strategies**:
   - **Risk Parity**: Equalize risk contributions $\sigma_i(w) = \sigma(w)/d$
   - **Minimum Variance**: $\min w^\top \Sigma w$ s.t. $w^\top \mathbf{1} = 1, w \geq 0$
   - **Mean-Variance**: $\min w^\top \Sigma w$ s.t. $w^\top \mu \geq \alpha$, $w^\top \mathbf{1} = 1, w \geq 0$ ($\alpha = 10\%$ annualized)
3. **Rebalancing**: Annual (February 1st each year) — ACC clusters change frequently, so less frequent rebalancing is better

---

## Empirical Results (from paper)

The paper backtests on S&P 500 (Feb 2001 - Jan 2020, ~19 years) with 500-day lookback windows. Key findings:

| Strategy | Ending VAMI | Sharpe | Max DD | Ann. Return |
|----------|------------|--------|--------|-------------|
| ACC (Risk Parity) | 8206.92 | 0.79 | 44.08% | 11.75% |
| ACC (Min Var) | 7299.02 | 0.84 | 32.45% | 11.06% |
| ACC (Mean-Var) | 7575.26 | 0.86 | 31.16% | 11.28% |
| SPY (benchmark) | 3442.53 | 0.36 | 55.25% | 6.74% |

ACC significantly outperforms SPY and sector ETF benchmarks across all metrics.

---

## Implementation Notes

1. **$\rho^{-1/2}$ computation**: Use eigenvalue decomposition $\rho = Q\Lambda Q^\top$, then $\rho^{-1/2} = Q \Lambda^{-1/2} Q^\top$
2. **Heavy-tail estimation**: Linear regression on log-log plot of tail quantiles (Gardes & Girard, 2008)
3. **Risk parity solver**: Non-linear optimization — can use iterative algorithms (Maillard et al., 2010)
4. **No short selling**: All optimization constraints include $w \geq 0$
5. **Singleton clusters**: PARTITION naturally produces them when no pair has CORD below $\varepsilon$
6. **Data filtering** (paper): Remove stocks with <5yr history or >5% missing data; interpolate missing prices linearly

---

## Deviations & Interpretations

- The paper uses Compustat/WRDS data (1996-2020). Our implementation uses yfinance 5-year data (~2021-2026), so the time period differs.
- The paper filters to 465-488 eligible stocks after data cleaning. With 503 S&P 500 tickers from yfinance, we use all available data with simple handling of missing values.
- The paper uses $n=500$ lookback with ~20 years of data (many rebalancing points). Our 5-year window limits the number of monthly/annual rebalancing points.
- Risk parity optimization uses the CCAP (Cyclical Coordinate Ascent Projection) or similar iterative solver rather than the exact formulation in the paper.
