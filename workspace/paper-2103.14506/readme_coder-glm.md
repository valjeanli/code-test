# Asset Selection via Correlation Blockmodel Clustering

**Paper:** arXiv:2103.14506v2  
**Authors:** Wenpin Tang, Xiao Xu, Xun Yu Zhou  
**Date:** August 16, 2021 (v2)  
**Venue:** (Pre-print / arXiv)  
**Category:** q-fin.PM (Portfolio Management)  

---

## 1. Problem Statement & Motivation

The paper addresses **asset selection for portfolio diversification**: given a universe of d financial assets (e.g., S&P 500), select a much smaller subset (15–25 stocks) that achieves the same level of diversification as the full universe. This is the first stage of a two-stage decomposition of Markowitz's mean-variance optimization:

1. **Asset selection** — choose which stocks to include
2. **Asset allocation** — decide how much to invest in each

The key insight is that if assets in the same cluster are highly correlated (Criterion 1) and have the same correlations with all assets outside the cluster (Criterion 2), then any asset in the cluster can substitute for any other in a portfolio context. Selecting one representative from each cluster yields a diversified portfolio with far fewer stocks.

---

## 2. Core Criteria

- **Criterion 1:** Financial assets in the same group have high correlations.
- **Criterion 2:** Financial assets in the same group have similar correlations with all other assets.

Criterion 1 ensures cluster cohesion; Criterion 2 ensures interchangeability within clusters, which directly supports portfolio diversification.

---

## 3. Correlation Blockmodel (Equation 1)

The standardized returns are modeled as:

$$X_i^* = F_{z(i)} + U_i, \quad i \in [d]$$

where:
- $F = (F_1, \ldots, F_K)$ are latent factors with $\mathbb{E}(F_k) = 0$
- $U = (U_1, \ldots, U_d)$ are idiosyncratic fluctuations with $\mathbb{E}(U_i) = 0$, $\mathrm{Cov}(U_i, U_j) = 0$ for $i \neq j$, and $\mathrm{Cov}(F_k, U_i) = 0$
- $z: [d] \to [K]$ is the membership assignment function
- $\sigma_k^2 = \mathrm{Var}(F_k)$ is the factor variance for group $k$

The correlation matrix decomposes as:

$$\rho = Z \Sigma_F Z^\top + \Sigma_U \quad \text{(Equation 2)}$$

where $Z$ is the $d \times K$ membership matrix and $\Sigma_U$ is diagonal.

---

## 4. Theorem 1 — Unique Coarsest Partition

The coarsest partition $G^*$ is unique and defined by:

$$i \overset{G^*}{\sim} j \iff \max_{l \neq i,j} |\rho_{il} - \rho_{jl}| = 0 \quad \text{(Equation 3)}$$

This provides the key dissimilarity measure: **CORD** (Correlation Difference):

$$\widehat{\mathrm{CORD}}(i, j) := \max_{l \neq i,j} |\hat{\rho}_{il} - \hat{\rho}_{jl}|, \quad i, j \in [d] \quad \text{(Equation 7)}$$

---

## 5. Theorem 2 — Optimal Asset Selection

Under the blockmodel with partition $G^* = \{G_1, \ldots, G_K\}$, select one asset $J(k)$ from each cluster. The minimum-variance portfolio selects the asset with the **lowest variance** in each cluster:

$$J^*(k) = \arg\min_{j \in G_k} \mathrm{Var}(X_j) \quad \forall k = 1, \ldots, K \quad \text{(Equation 5)}$$

---

## 6. The PARTITION Procedure (Procedure 1)

```python
def PARTITION(D, epsilon):
    """D: dissimilarity matrix (d x d), epsilon: threshold > 0"""
    S = set(range(d))   # remaining unassigned assets
    clusters = []
    while S is not empty:
        if |S| == 1:
            cluster = S
        else:
            (i_l, j_l) = argmin_{i,j in S, i≠j} D(i, j)
            if D(i_l, j_l) > epsilon:
                cluster = {i_l}  # singleton
            else:
                cluster = {k in S : min(D(i_l, k), D(j_l, k)) <= epsilon}
        clusters.append(cluster)
        S = S - cluster
    return clusters
```

**Key property:** Does not require K as input; the number of clusters is determined by epsilon.

---

## 7. Heavy-Tailedness Estimation (Section 2.4)

The threshold ε depends on the tail parameter α and constant L from Assumption 1 (α-sub-exponential distribution). Estimation uses the tail quantile method from Gardes & Girard (2008):

For each asset $i$:
1. Compute $Y_i = |(\hat{\rho}^{-1/2} X^*)_i|$, the absolute decorrelated standardized returns
2. Sort: $Y_i[1] \leq Y_i[2] \leq \ldots \leq Y_i[n]$
3. Use the top k = n/4 observations
4. Regress $\log Y_i[n-j]$ against $\log\log(2n/j)$ for $1 \leq j \leq k$
5. Slope $s$ → $\alpha_i = 1/s$, intercept $a$ → $L_i = \exp(a)$

Then: $\alpha^* = \min_i \alpha_i$, $L^* = \max_i L_i$

---

## 8. Complete ACC Algorithm (Algorithm 1)

```python
def ACC(X, a=0.1, b=10, ng=100, U=(15, 25), k_ratio=0.25):
    """
    X: returns matrix (n × d)
    a, b: search range multipliers
    ng: number of grid points
    U: (min_clusters, max_clusters) range
    k_ratio: fraction of observations for tail estimation
    """
    n, d = X.shape
    
    # Step 1: Standardize returns
    X_star = (X - mean(X, axis=0)) / std(X, axis=0)
    
    # Step 2: Compute sample correlation matrix
    rho_hat = (1/(n-1)) * X_star.T @ X_star
    
    # Step 3: Compute CORD dissimilarity matrix
    CORD = zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i != j:
                CORD[i,j] = max_{l ≠ i,j} |rho_hat[i,l] - rho_hat[j,l]|
    # Set diagonal to infinity (or handle separately)
    
    # Step 4: Estimate alpha and L
    k = int(n * k_ratio)
    rho_hat_inv_sqrt = matrix_sqrt_inv(rho_hat)
    decorrelated = X_star @ rho_hat_inv_sqrt.T
    alphas = []
    Ls = []
    for i in range(d):
        Y_i = abs(decorrelated[:, i])
        Y_sorted = sort(Y_i)
        # Linear regression on tail
        y = log(Y_sorted[n-k:n-1])
        x = log(log(2*n / arange(1, k+1)))
        slope, intercept = linregress(x, y)
        alpha_i = 1 / slope
        L_i = exp(intercept)
        alphas.append(alpha_i)
        Ls.append(L_i)
    alpha = min(alphas)
    L = max(Ls)
    
    # Step 5: Determine search range for epsilon
    if n > (log(d)) ** (4/(alpha - 1)):
        T_low = a * L^2 * sqrt(log(d) / n)
        T_high = b * L^2 * sqrt(log(d) / n)
    else:
        T_low = a * L^2 * (log(d))**(2/alpha) / n
        T_high = b * L^2 * (log(d))**(2/alpha) / n
    T_high = min(T_high, 2.0)  # Cap at 2
    
    # Step 6: Grid search over epsilon
    epsilons = linspace(T_low, T_high, ng)
    best_rho_ave = -inf
    best_eps = epsilons[0]
    for eps in epsilons:
        clusters = PARTITION(CORD, eps)
        K = len(clusters)
        if K < U[0] or K > U[1]:
            continue
        # Compute intra-cluster correlation
        same_cluster = [(i,j) for each cluster for i in cluster for j in cluster if i < j]
        sum_corr = sum(rho_hat[i,j] for i,j in same_cluster)
        count = len(same_cluster)
        rho_ave = sum_corr / count if count > 0 else -inf
        if rho_ave > best_rho_ave:
            best_rho_ave = rho_ave
            best_eps = eps
    
    # Step 7: Return final clusters
    return PARTITION(CORD, best_eps)
```

---

## 9. Asset Selection (Per Theorem 2)

From each cluster, select the asset with the **lowest variance** (sample variance of daily returns in the backward-looking window).

---

## 10. Portfolio Allocation Strategies

### 10.1 Risk Parity (Equations 17–20)

Equalize risk contributions: $\sigma_i(w) = \sigma(w)/d$ for all $i$, where:

$$\sigma(w) = \sqrt{w^\top \Sigma w}, \quad \sigma_i(w) = \frac{w_i (\Sigma w)_i}{\sigma(w)}$$

Solved via nonlinear optimization:

$$\min_w \sum_{i=1}^{d} \left(w_i - \frac{\sigma(w)^2}{d \cdot (\Sigma w)_i}\right)^2 \quad \text{s.t. } w^\top \mathbf{1} = 1,\ w \geq 0$$

### 10.2 Minimum Variance (Equation 22)

$$\min_w w^\top \Sigma w \quad \text{s.t. } w^\top \mathbf{1} = 1,\ w \geq 0$$

### 10.3 Mean-Variance (Equation 21)

$$\min_w w^\top \Sigma w \quad \text{s.t. } w^\top \mu \geq \alpha,\ w^\top \mathbf{1} = 1,\ w \geq 0$$

where $\alpha$ is the target annualized return (set to 10% in the paper) and $\mu$ is the vector of mean daily returns.

---

## 11. Empirical Setup (Paper's Parameters)

| Parameter | Value | Notes |
|-----------|-------|-------|
| n (lookback window) | 500 trading days | ~2 years |
| k (tail observations) | n/4 = 125 | For alpha/L estimation |
| Search range multipliers | a=0.1, b=10 | Rule 1 |
| Number of grids | ng = 100 | Grid search granularity |
| Cluster count range U | [15, 25] | Rule 3 |
| Epsilon upper bound cap | 2.0 | Maximum CORD value is 2 |
| Rebalancing frequency | Annual | First trading day of February |
| Minimum history | 5 years | Stocks with less are discarded |
| Max missing data | 5% | Stocks exceeding this are discarded |
| Same-company handling | Keep longest history class | e.g., GOOG vs GOOGL |

---

## 12. Key Results

- ACC portfolios outperform SPY (S&P 500 ETF) significantly in Sharpe, Sortino, and Calmar ratios
- ACC with risk parity + annual rebalancing: Sharpe = 0.79 (vs SPY Sharpe = 0.36)
- ACC clusters are less similar to GICS sectors than k-medoids clusters → they uncover "under-the-radar" stocks
- ACC performs best with infrequent (annual) rebalancing; performance degrades with more frequent rebalancing
- Estimated α values for S&P 500 data range from ~0.45 to ~0.65, confirming heavy tails

---

## 13. Implementation Notes & Deviations

1. **Correlation matrix near-singularity:** In practice, the sample correlation matrix of 500 stocks with 500 observations may be ill-conditioned. We use regularization (adding a small ridge) to ensure invertibility for `ρ^{-1/2}`.

2. **CORD computation:** The naive O(d³) computation can be vectorized: for each pair (i,j), compute `max_l |ρ[i,:] - ρ[j,:]|` excluding l=i and l=j.

3. **Grid search:** The epsilon range depends on estimated α and L; if n >> log(d)^{4/(α-1)}, we use the simpler √(log(d)/n) formula.

4. **Asset selection criterion:** We use lowest sample variance (not Sharpe ratio), consistent with Theorem 2 and the paper's empirical approach.

5. **Portfolio backtesting:** We implement all three allocation strategies with annual rebalancing, using the same backward-looking 500-day window.

6. **Data source deviation:** The paper uses Compustat/WRDS data (1996-2020). Our implementation uses the S&P 500 data from the parent task (yfinance, 2021-2026), which is a different time period and source. Results will differ but the algorithm is identical.

7. **k-medoids benchmark:** We implement k-medoids using sklearn_extra or a manual implementation using correlation-based distance D(a,b) = √(2(1-ρ_ab)).

---

## 14. Complexity Analysis

- **CORD matrix:** O(nd²) — computing sample correlation matrix plus pairwise CORD
- **PARTITION:** O(d³) — finding argmin pairs and scanning
- **Grid search:** O(ng × d³) — running PARTITION for each grid point
- **Total ACC:** O(nd² + d³ × ng) per time step
- **Paper theorem:** O(nd² + d³) without grid search overhead

---

## 15. References

- Tang, W., Xu, X., & Zhou, X. Y. (2021). Asset Selection via Correlation Blockmodel Clustering. arXiv:2103.14506v2.
- Bunea, F., Giraud, C., & Luo, X. (2016). Minimax optimal variable clustering in G-models via CORD. arXiv:1508.01939v2.
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77–91.
- Maillard, S., Roncalli, T., & Teïletche, J. (2010). The Properties of Equally Weighted Risk Contribution Portfolios. Journal of Portfolio Management, 36(4), 60–70.