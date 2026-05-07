# Paper 2103.14506: Asset Selection via Correlation Blockmodel Clustering

**Authors:** Wenpin Tang, Xiao Xu, Xun Yu Zhou  
**Year:** 2021  
**Venue:** arXiv (q-fin.PM), Columbia University  
**Link:** https://arxiv.org/abs/2103.14506

---

## Problem Statement & Motivation

Modern portfolio theory (Markowitz 1952) requires investing across many assets for diversification, but practical constraints limit how many stocks an investor can hold. The paper asks: **How can we select a much smaller subset of stocks that achieves a sufficient level of diversification?**

Key insight: Since diversification depends on asset **correlations**, clustering based directly on correlation structure is more natural than sector-based classification.

---

## Two Core Criteria

**Criterion 1:** Assets in the same cluster should have **high correlations** with each other.

**Criterion 2:** Assets in the same cluster should have **the same correlations with all other assets** (interchangeability).

Criterion 2 is the novel contribution — it ensures that any two assets in a cluster are substitutable in terms of their correlation footprint with the rest of the universe, making the choice of representative assets simple (pick by idiosyncratic metrics like Sharpe ratio or volatility).

---

## Correlation Blockmodel

### Model (Eq. 1)
Standardized returns are decomposed as:

```
X*_i = F_{z(i)} + U_i,   i in [d]
```

Where:
- `X*_i = (X_i - E(X_i)) / sqrt(Var(X_i))` — standardized return of asset i
- `F = (F_1, ..., F_K)` — latent factors with `E(F_k) = 0`
- `U = (U_1, ..., U_d)` — idiosyncratic noise, `E(U_i) = 0`, `Cov(U_i, U_j) = 0` for i ≠ j
- `z(i)` — cluster membership of asset i
- `Cov(F_k, U_i) = 0` for all k, i

### Correlation Matrix Decomposition (Eq. 2)
```
ρ = Z Σ_F Z^T + Σ_U
```
Where:
- `ρ = E(X* X*^T)` — population correlation matrix
- `Z` — membership matrix (binary indicator)
- `Σ_F = E(F F^T)` — factor covariance
- `Σ_U = E(U U^T)` — diagonal noise covariance

### Identifiability (Theorem 1)
The **coarsest partition** G* is unique and defined by:
```
i ~_G* j  iff  max_{l≠i,j} |ρ_{il} - ρ_{jl}| = 0
```

---

## CORD Dissimilarity Measure (Eq. 4)

**Correlation Difference (CORD):**
```
CORD(i, j) := max_{l≠i,j} |ρ_{il} - ρ_{jl}|
```

This measures how differently assets i and j correlate with every other asset. CORD(i,j) = 0 means i and j are perfectly interchangeable in a portfolio context.

**Distance metric:** `D(a,b) = sqrt(2(1 - ρ_{ab}))` (Mantegna 1999)

---

## The PARTITION Procedure (Algorithm 2)

```
Input: CORD matrix, threshold ε
Output: Partition G = {G_1, ..., G_K}

1. S ← {1, ..., d}     (set of unassigned assets)
2. G ← {}               (empty partition)
3. WHILE S is not empty:
   a. Pick i_t from S (first element or random)
   b. Find j_t = argmin_{j∈S, j≠i_t} CORD(i_t, j)
   c. IF CORD(i_t, j_t) > ε:
        G_t ← {i_t}              (singleton cluster)
   d. ELSE:
        G_t ← {k ∈ S : min(CORD(i_t, k), CORD(j_t, k)) ≤ ε}
   e. G ← G ∪ {G_t}
   f. S ← S \ G_t
4. RETURN G
```

**Key idea:** For each seed asset, find its most similar neighbor. If they're close enough (≤ ε), pull in all assets close to either of them. Otherwise, the seed is a singleton.

---

## The ACC Algorithm (Algorithm 1)

### Stage 1: Preparation
1. Compute daily returns from price data
2. Standardize returns: `X*_i = (X_i - mean(X_i)) / std(X_i)`
3. Compute sample correlation matrix `ρ̂`
4. Compute sample CORD matrix: `CORD̂(i,j) = max_{l≠i,j} |ρ̂_{il} - ρ̂_{jl}|`
5. Estimate tail index α using Hill's estimator on standardized returns

### Stage 2: Grid Search
1. Set grid bounds:
   - Lower: `ε_min = c₁ √(log d / n)`
   - Upper: `ε_max = 0.5` (or based on CORD distribution)
   - Number of grid points: `n_g` (typically 10-20)
2. For each ε in the grid:
   - Run PARTITION(ε) → G_ε
   - Record number of clusters K_ε
3. Select ε* that gives "reasonable" K (typically 15-25 clusters for S&P 500)
   - Use Criterion 1 (within-cluster correlation) for cross-validation

### Complexity
Overall: **O(d²(n + d))** where d = number of assets, n = number of time periods

---

## Portfolio Construction

### Risk Parity (Eq. 17-20)
Equalize weighted marginal risk contribution:
```
σ_i(w) = w_i * (Σw)_i / σ(w)
Target: σ_i(w) = σ(w) / d  for all i
Optimization: min Σ (w_i - σ(w)² / (d * (Σw)_i))²
Subject to: w^T 1 = 1, w ≥ 0
```

### Minimum Variance (Eq. 22)
```
min w^T Σ w
Subject to: w^T 1 = 1, w ≥ 0
```

### Mean-Variance (Eq. 21)
```
min w^T Σ w
Subject to: w^T μ ≥ α, w^T 1 = 1, w ≥ 0
```
Where α = target return (set to 10% in paper)

---

## Implementation Notes

1. **Data Requirements:**
   - Daily OHLCV price data for S&P 500 stocks
   - Minimum 500 trading days lookback window
   - No missing values (clean/forward-fill)

2. **Standardization:**
   - Returns must be standardized (zero mean, unit variance per asset)
   - Critical for CORD computation

3. **Correlation Matrix:**
   - Use Pearson correlation of standardized returns
   - Sample correlation: `ρ̂ = (1/n) X*^T X*`

4. **Hill Estimator for Tail Index:**
   ```
   Sort |X*_i| in decreasing order: Y_(1) ≥ Y_(2) ≥ ... ≥ Y_(n)
   α̂ = (1/k) Σ_{i=1}^{k} log(Y_(i) / Y_(k+1))
   ```
   Where k is a tuning parameter (typically k = n^0.6 or similar)

5. **Grid Search Tuning:**
   - Paper uses n_g = 20 grid points
   - Select ε where cluster count stabilizes or is in desired range
   - Cross-validate with within-cluster average correlation

6. **Portfolio Backtesting:**
   - Rolling window: 500 trading days
   - Rebalancing: Quarterly or annually
   - Equal selection from each cluster (one stock per cluster)
   - Representative stock: lowest volatility or highest Sharpe ratio in cluster

7. **Edge Cases:**
   - Single-asset clusters (assets too different from all others)
   - Very large clusters (assets very similar) — may need to split
   - Heavy-tailed data: use robust correlation (Kendall/Spearman) as alternative

8. **Paper's S&P 500 Experiment Details:**
   - Period: 2002-01-01 to 2020-09-30
   - Lookback: 500 trading days
   - Rebalancing: Quarterly (21 trading days) and annually
   - Select 15-25 stocks (one per cluster)
   - Three allocation strategies compared
   - Benchmark: SPY ETF

---

## Key Theoretical Results

**Theorem 1:** The coarsest partition G* is unique and identifiable.

**Theorem 2:** Under the blockmodel, any representative portfolio (one stock per cluster) has the same correlation structure, making minimum-variance portfolios equivalent up to volatility scaling.

**Theorem 3:** With high probability, PARTITION recovers G* when `ε` satisfies `τ ≤ ε < Δ - τ`, where τ is the sampling error and Δ = min_{i~̸_G* j} CORD(i,j).

**Theorem 4:** ACC runs in O(d²(n + d)) time.

---

## Benchmark Datasets

- **S&P 500 stocks**: Daily OHLCV from 2002 to 2020
- **Benchmark portfolio**: SPY (S&P 500 ETF)
- **Comparison methods:**
  - k-medoids clustering (K=20)
  - S&P sector/industry classification
  - S&P 500 sector ETFs (11 sectors)

---

## Deviations from Paper (Our Implementation)

1. **Data period**: We use 2021-05-10 to 2026-05-06 (5 years of recent data) instead of 2002-2020
2. **Lookback window**: 252 trading days (1 year) instead of 500, to allow more backtest history
3. **Rebalancing**: Monthly instead of quarterly/annual for more granular analysis
4. **Cluster selection**: We use volatility-based representative selection (simplest approach per paper)
5. **Grid search**: We use a simplified heuristic for ε selection based on cluster count range
