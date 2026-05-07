# Asset Selection via Correlation Blockmodel Clustering

**Paper**: arXiv 2103.14506
**Title**: Asset Selection via Correlation Blockmodel Clustering
**Authors**: Wenpin Tang, Xiao Xu, Xun Yu Zhou (Columbia University)
**Date**: March 2021 (v1), August 2021 (v2)
**Venue**: Quantitative Finance > Portfolio Management
**Pages**: 46 pages, 9 figures, 8 tables

---

## 1. Problem Statement & Motivation

The modern portfolio theory (Markowitz) requires investing in ALL available assets for diversification. For S&P 500 this means 500 stocks — impossible for small investors. The paper asks:

> **How can we select a much smaller subset of stocks that achieves sufficient diversification?**

Key insight: Decompose Markowitz into two stages:
1. **Asset Selection** — pick a small diverse subset via clustering
2. **Asset Allocation** — apply mean-variance to the subset

## 2. Two Clustering Criteria

**Criterion 1**: Assets in the same cluster have HIGH correlations with each other.
**Criterion 2**: Assets in the same cluster have SIMILAR correlations with all other assets (interchangeability).

Criterion 2 is the novel contribution — it means any two assets in the same group are interchangeable for portfolio construction (in terms of correlation structure). You can then pick the best one based on idiosyncratic characteristics (e.g., lowest volatility).

## 3. Correlation Blockmodel (Equation 1)

The standardized returns X*_i are modeled as:

```
X*_i = F_{z(i)} + U_i    for i ∈ [d]
```

Where:
- `F_k` are K latent factors with E[F_k] = 0
- `U_i` are idiosyncratic fluctuations with E[U_i] = 0, Cov(U_i, U_j) = 0 for i ≠ j, Cov(F_k, U_i) = 0

The correlation matrix decomposes as (Equation 2):
```
ρ = Z Σ_F Z^T + Σ_U
```
Where Z is the membership matrix, Σ_F is factor covariance, Σ_U is diagonal.

**Theorem 1**: There exists a unique coarsest partition G* defined by the equivalence relation (Equation 3):
```
i ~ j  iff  max_{l ≠ i,j} |ρ_il - ρ_jl| = 0
```

## 4. Dissimilarity Measure: CORD (Equation 4)

```
CORD(i, j) = max_{l ≠ i,j} |ρ_il - ρ_jl|
```

This quantifies how different two assets are in terms of their correlations with all other assets. CORD(i,j) = 0 means i and j have identical correlation profiles.

## 5. PARTITION Procedure (Procedure 1)

```
Input: Dissimilarity matrix D (d×d), threshold ε > 0
Output: Partition Ĝ = {Ĝ_1, Ĝ_2, ...}

Initialization: S = {1,...,d}  (unassigned assets), l = 0

while S ≠ ∅:
    l = l + 1
    if |S| == 1:
        Ĝ_l = S                    # singleton cluster
    else:
        (i_l, j_l) = argmin_{i,j∈S, i≠j} D(i,j)    # most similar pair
        if D(i_l, j_l) > ε:
            Ĝ_l = {i_l}            # singleton (core asset alone)
        else:
            Ĝ_l = {k ∈ S : min(D(i_l,k), D(j_l,k)) ≤ ε}  # all similar to either core
    S = S \ Ĝ_l

return Ĝ
```

**Key property**: Does NOT require the number of clusters K as input — K is determined by ε.

## 6. Sample Versions (Equations 6-7)

Sample correlation matrix (n observations):
```
ρ̂ = (1/(n-1)) (X*)^T X*
```

Sample CORD:
```
CÔRD(i,j) = max_{l ≠ i,j} |ρ̂_il - ρ̂_jl|
```

## 7. Statistical Guarantee (Theorem 3)

Under Assumption 1 (α-sub-exponential returns, α ∈ (0,2]):

If mini_{G*≁j} CORD(i,j) > ε and:
```
ε ≥ 2L² [c₁ √(log d / n) + c₂ (log d)^(2/α) / n]
```
then PARTITION(CÔRD, ε) recovers G* with probability 1 - 4/d.

For d=500, this is 99.2% probability.

## 8. Tuning ε — Three Rules

### Rule 1: Search Range
If n > (log d)^(4/α - 1):
```
T = [a, b] × L² √(log d / n)
```
else:
```
T = [a, b] × L² (log d)^(2/α) / n
```
Paper uses a=0.1, b=10. Cap upper bound at 2.0.

### Rule 2: Maximize Intra-Cluster Correlation (Equation 9)
```
ρ̂_ave(ε) = Σ_{i<j} 1[i ~_{Ĝ_ε} j] · ρ̂_ij  /  Σ_{i<j} 1[i ~_{Ĝ_ε} j]
```
Choose ε that maximizes this over the search range T.

### Rule 3: Cluster Count Constraint
```
ε* = argmax_{ε∈T, |Ĝ_ε|∈U} ρ̂_ave(ε)
```
Where U is the desired range for number of clusters (paper uses [15, 25]).

## 9. Heavy-Tailedness Estimation (Section 2.4)

**Assumption 1**: ρ^(-1/2) X* is α-sub-exponential with constant L.

Estimation via tail regression:
1. Compute Y_r = |(ρ̂^(-1/2) X*)_r| for each asset r
2. Sort: Y_r(1) ≤ Y_r(2) ≤ ... ≤ Y_r(n)
3. Take k = n/4 largest observations
4. Regress log Y_r(n-j) against log log(2n/j) for j=1..k
5. Slope = 1/α_r, intercept = log L_r
6. α = min_r α_r, L = max_r L_r

The quantile function: q(p) = L [log(2/(1-p))]^(1/α)
Log-linear relationship: log q(p) = (1/α) log log(2/(1-p)) + log L

## 10. Full ACC Algorithm (Algorithm 1)

```
ACC(X, a=0.1, b=10, ng=100, U=[15,25], k=n/4):
    1. Standardize returns: X* = (X - mean(X)) / std(X)  [column-wise]
    2. Compute sample correlation: ρ̂ = (1/(n-1)) (X*)^T X*
    3. Compute CÔRD matrix: CÔRD(i,j) = max_{l≠i,j} |ρ̂_il - ρ̂_jl|
    4. For each asset i: estimate α_i, L_i via tail regression on |(ρ̂^(-1/2) X*)_i|
       α = min(α_i), L = max(L_i)
    5. Determine search range T via Rule 1
    6. Grid search: for each ε in ng evenly-spaced points on T:
       a. Ĝ_ε = PARTITION(CÔRD, ε)
       b. If |Ĝ_ε| ∈ U: compute ρ̂_ave(ε) via Equation 9
       c. Else: ρ̂_ave(ε) = -∞
    7. ε* = argmax ρ̂_ave(ε)
    8. Return PARTITION(CÔRD, ε*)
```

**Complexity**: O(n·d² + d³) arithmetic operations (Theorem 4)

## 11. Portfolio Construction (Section 3.3)

1. Select the **lowest volatility** stock from each cluster (Theorem 2 justification)
2. Volatility = sample std of daily returns in lookback window
3. Rebalance annually (1st trading day of February)
4. Three allocation strategies: risk parity, minimum variance, mean-variance (10% target)

## 12. Key Assumptions & Constraints

- Returns are i.i.d. across time
- Assumption 1: α-sub-exponential distribution (α ∈ (0,2])
  - α=2: sub-Gaussian, α=1: sub-exponential, α<1: heavy-tailed
  - Financial data typically has α ≈ 0.45-0.65
- Correlation matrix ρ is non-singular (no redundant securities)
- Minimum 5 years of history per stock
- Maximum 5% missing data

## 13. Empirical Results (Paper)

- **Backtest**: Feb 2001 – Jan 2020, S&P 500 universe
- **Lookback**: 500 trading days
- **Rebalance**: Annual (February)
- **ACC clusters**: 15-25 clusters per period
- **Results**: ACC significantly outperforms SPY ETF, GICS-based, k-medoids, and sector ETF portfolios across all 3 allocation strategies (risk parity, min-variance, mean-variance)
- ACC Sharpe: 0.79 vs SPY: 0.36 (risk parity)
- Max drawdown recovery: 261 days vs SPY: 774 days

## 14. Implementation Notes

### Data Preparation
- Use closing prices, compute daily log returns (or simple returns as in paper)
- Standardize: subtract mean, divide by std (column-wise)
- Filter stocks with < 500 data points or > 5% missing data

### CORD Matrix Computation
- For d assets: O(d³) — compute all pairwise differences
- Can be optimized by computing difference between each row of ρ̂
- Requires careful handling of the "l ≠ i,j" constraint

### PARTITION Procedure
- In each iteration, find global minimum of D(i,j) for i,j in remaining set S
- Add all assets with distance ≤ ε from EITHER core i_l or j_l
- Continues until all assets assigned

### α Estimation
- Need to invert ρ̂: compute ρ̂^(-1/2) using eigendecomposition
- For ill-conditioned ρ̂: use pseudoinverse or regularization
- k = n/4 for the tail (paper uses 125 for n=500)
- Linear regression in log-log space

### Grid Search
- ng = 100 grid points in range T
- Cap upper bound at 2.0
- Can parallelize but not necessary for d~500

### Performance Considerations
- ρ̂^(-1/2) computation: O(d³) via eigendecomposition → ~125M ops for d=500 (fast)
- CORD matrix: O(d³) → compute pairwise row differences efficiently
- PARTITION: O(K·d²) where K = number of clusters (~20)
- Grid search: 100 × O(K·d²) → manageable

## 15. Edge Cases

- **Singular correlation matrix**: Regularize with small diagonal term (λI)
- **All CORD values > ε**: Every asset becomes its own cluster → K = d (rejected by Rule 3)
- **All CORD values ≤ ε**: All assets in one cluster → K = 1 (rejected by Rule 3)
- **Ties in CORD**: Choose first pair arbitrarily (deterministic ordering)
- **Multiple ε giving same ρ̂_ave**: Choose smallest ε (finer partition as tiebreaker)
- **NaN correlations**: Drop those asset pairs, set CORD to large value
- **n > d constraint**: n≈1250, d≈500 → n > d, OK
