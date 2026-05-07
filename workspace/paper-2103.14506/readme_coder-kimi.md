# Asset Selection via Correlation Blockmodel Clustering

**Paper:** Tang, W., Xu, X., & Zhou, X. Y. (2021). "Asset Selection via Correlation Blockmodel Clustering." *arXiv:2103.14506 [q-fin.PM]*.

---

## 1. Problem Statement & Motivation

The paper addresses a fundamental question in portfolio theory: **how to select a small subset of stocks that approximates the diversification level of the entire universe**. 

- Naive Markowitz mean-variance optimization on all S&P 500 stocks is impractical for small investors.
- Simply adding sparsity constraints (e.g., L1 regularization) does not guarantee diversification.
- The authors propose a **data-driven clustering approach** based purely on return correlations, without relying on sector/industry taxonomies.

### Key Insight
Cluster stocks such that:
- **Criterion 1:** Assets in the same cluster are highly correlated with each other.
- **Criterion 2:** Assets in the same cluster have the **same correlations with all other assets** in the universe.

Criterion 2 ensures interchangeability: any asset in a cluster can be swapped for another without changing the portfolio's correlation structure with the rest of the market.

---

## 2. Correlation Blockmodel

### 2.1 Model Setup

Let $X = (X_1, \ldots, X_d)^\top$ be asset returns. Let $X_i^* = (X_i - \mathbb{E}[X_i]) / \sqrt{\text{Var}(X_i)}$ be standardized returns.

The **correlation blockmodel** assumes:

$$X_i^* = F_{z(i)} + U_i, \quad i \in [d]$$

where:
- $F = (F_1, \ldots, F_K)$ are latent factors with $\mathbb{E}[F_k] = 0$.
- $U = (U_1, \ldots, U_d)$ are idiosyncratic noises with $\mathbb{E}[U_i] = 0$, $\text{Cov}(U_i, U_j) = 0$ for $i \neq j$, and $\text{Cov}(F_k, U_i) = 0$.
- $z: [d] \to [K]$ is the unknown membership map.

Under this model, for $i \sim j$ (same cluster) and $l \neq i, j$:
$$\text{Corr}(X_i, X_l) = \mathbb{E}[F_{z(i)} F_{z(l)}] = \mathbb{E}[F_{z(j)} F_{z(l)}] = \text{Corr}(X_j, X_l)$$

The correlation matrix is:
$$\rho = Z \Sigma_F Z^\top + \Sigma_U$$

where $Z$ is the membership matrix, $\Sigma_U$ is diagonal.

### 2.2 Identifiability

**Theorem 1:** There exists a unique **coarsest partition** $G^*$ such that $\rho = Z \Pi Z^\top + \Gamma$ for some $\Pi$ and diagonal $\Gamma$. The equivalence relation is:

$$i \sim_{G^*} j \iff \max_{l \neq i,j} |\rho_{il} - \rho_{jl}| = 0$$

This defines the target clusters we want to recover from data.

---

## 3. Core Algorithm: ACC

### 3.1 Dissimilarity Measure: CORD

The **Correlation Difference (CORD)** between assets $i$ and $j$ is:

$$\text{CORD}(i, j) := \max_{l \neq i,j} |\rho_{il} - \rho_{jl}|$$

From Theorem 1, $i$ and $j$ are in the same cluster iff $\text{CORD}(i, j) = 0$.

From data, we use the **sample CORD**:

$$\widehat{\text{CORD}}(i, j) := \max_{l \neq i,j} |\hat{\rho}_{il} - \hat{\rho}_{jl}|$$

where $\hat{\rho} = \frac{1}{n-1} (X^*)^\top X^*$ is the sample correlation matrix.

### 3.2 PARTITION Procedure

**Inputs:** Dissimilarity matrix $D$, threshold $\varepsilon > 0$

```
PARTITION(D, epsilon):
    S <- [d]
    clusters <- []
    while S is not empty:
        if |S| == 1:
            clusters.append(S)
            S <- S \ S
        else:
            (i, j) <- argmin_{i,j in S, i!=j} D(i, j)
            if D(i, j) > epsilon:
                clusters.append({i})
                S <- S \ {i}
            else:
                G <- {k in S : min(D(i, k), D(j, k)) <= epsilon}
                clusters.append(G)
                S <- S \ G
    return clusters
```

**Key properties:**
- Does **not** require the number of clusters $K$ as input; $K$ is determined by $\varepsilon$.
- Iteratively peels off one cluster at a time, starting from the most similar pair.

### 3.3 Heavy-Tailedness Estimation

Financial returns are heavy-tailed. The paper assumes $\rho^{-1/2} X^*$ is $\alpha$-sub-exponential.

To estimate $\alpha$ and $L$:
1. Whiten the data: $Y = |\rho^{-1/2} X^*| \in \mathbb{R}^{n \times d}$
2. For each asset $r$, sort the $n$ observations: $Y_{r,(1)} \leq \ldots \leq Y_{r,(n)}$
3. Take the largest $k$ observations (excluding the very largest).
4. Run linear regression:
   $$\log Y_{r,(n-j)} \sim \log \log(2n / j), \quad j = 1, \ldots, k$$
5. Slope $s_r \to \alpha_r = 1/s_r$, intercept $a_r \to L_r = \exp(a_r)$
6. Aggregate: $\alpha = \min_r \alpha_r$, $L = \max_r L_r$

This captures the heaviest tail across all assets.

### 3.4 Tuning the Threshold $\varepsilon$

**Rule 1 (Search Range):**
- If $n > (\log d)^{4/\alpha - 1}$: $T = [a, b] \times L^2 \sqrt{\frac{\log d}{n}}$
- Else: $T = [a, b] \times L^2 \frac{(\log d)^{2/\alpha}}{n}$
- Cap upper bound at 2.

**Rule 2 (Intra-cluster Correlation):**
For a partition $\hat{G}_\varepsilon$, define average intra-cluster correlation:

$$\hat{\rho}^{\text{ave}}_\varepsilon := \frac{\sum_{i<j} \mathbb{1}(i \sim_{\hat{G}_\varepsilon} j) \cdot \hat{\rho}_{ij}}{\sum_{i<j} \mathbb{1}(i \sim_{\hat{G}_\varepsilon} j)}$$

Choose $\varepsilon$ that maximizes this.

**Rule 3 (Cluster Count Regularization):**
Let $U$ be a user-defined range for the number of clusters (e.g., $[15, 25]$). Choose:

$$\varepsilon^* = \arg\max_{\varepsilon \in T,\; |\hat{G}_\varepsilon| \in U} \hat{\rho}^{\text{ave}}_\varepsilon$$

This prevents trivial solutions (all singletons or one giant cluster).

### 3.5 Complete ACC Algorithm

```
ACC(X, a, b, ng, U, k):
    Input: returns X in R^{n x d}, search range multipliers [a, b],
           number of grids ng, cluster count range U, tail obs k

    // Step 1: Standardize
    X* <- (X - mean(X)) / std(X)    // column-wise

    // Step 2: Sample correlation
    rho_hat <- (1/(n-1)) * (X*)^T X*

    // Step 3: Sample CORD
    CORD_hat(i,j) <- max_{l != i,j} |rho_hat_{i,l} - rho_hat_{j,l}|

    // Step 4: Estimate heavy-tailedness
    for r in [d]:
        Y_r <- |(rho_hat^{-1/2} X*)_r|
        Sort Y_r
        (s, a_int) <- LinearRegression(log Y_r[n-k:n-1] ~ log log(2n / [1:k]))
        alpha_r <- 1/s
        L_r <- exp(a_int)
    alpha <- min_r alpha_r
    L <- max_r L_r

    // Step 5: Determine search range
    if n > (log d)^{4/alpha - 1}:
        T <- [a, b] * L^2 * sqrt(log d / n)
    else:
        T <- [a, b] * L^2 * (log d)^{2/alpha} / n
    T.upper <- min(T.upper, 2)

    // Step 6: Grid search
    epsilons <- linspace(T.lower, T.upper, ng)
    best_eps <- None
    best_rho_ave <- -inf
    for eps in epsilons:
        G_eps <- PARTITION(CORD_hat, eps)
        K_eps <- |G_eps|
        if K_eps in U:
            rho_ave <- intra_cluster_correlation(G_eps, rho_hat)
            if rho_ave > best_rho_ave:
                best_rho_ave <- rho_ave
                best_eps <- eps

    // Step 7: Return best partition
    return PARTITION(CORD_hat, best_eps)
```

**Theorem 4:** The ACC algorithm requires at most $O(nd^2 + d^3)$ arithmetic operations.

### 3.6 Statistical Guarantee

**Theorem 3:** Under Assumption 1 (non-singular correlation, $\alpha$-sub-exponential), if:
$$\min_{i \not\sim_{G^*} j} \text{CORD}(i, j) > \varepsilon \geq 2L^2 \left( c_1 \sqrt{\frac{\log d}{n}} + c_2 \frac{(\log d)^{2/\alpha}}{n} \right)$$

then PARTITION with inputs $\widehat{\text{CORD}}$ and $\varepsilon$ outputs $\hat{G} = G^*$ with probability $1 - 4/d$.

For $d = 500$, this probability is $99.2\%$.

---

## 4. Key Assumptions & Constraints

| Assumption | Description |
|---|---|
| Non-singular $\rho$ | No redundant securities (no linear combination of other assets) |
| $\alpha$-sub-exponential | Heavy-tailed returns allowed; $\alpha \in (0, 2]$; $\alpha = 2$ is sub-Gaussian, $\alpha = 1$ is sub-exponential |
| i.i.d. observations | $X_1, \ldots, X_n$ are i.i.d. copies |
| Coarsest partition | We seek the coarsest (fewest clusters) partition satisfying the blockmodel |

**Practical parameters (from paper's empirical analysis):**
- $n = 500$ trading days lookback
- $a = 0.1$, $b = 10$ (search range multipliers)
- $n_g = 100$ grid points
- $U = [15, 25]$ cluster count range
- $k = n/4 = 125$ tail observations for $\alpha$ estimation
- Upper bound on $\varepsilon$ capped at 2 (since max CORD difference is 2)

---

## 5. Input / Output Specifications

### Input
- **Raw returns matrix** $X \in \mathbb{R}^{n \times d}$: $n$ observations (e.g., daily returns), $d$ assets.
- **Hyperparameters:** $a, b, n_g, U, k$.

### Output
- **Partition** $\hat{G} = \{\hat{G}_1, \ldots, \hat{G}_K\}$ where $\bigcup_k \hat{G}_k = [d]$ and clusters are disjoint.

### Post-Processing: Asset Selection
From each cluster $\hat{G}_k$, select one representative asset. The paper suggests (Theorem 2):
- For minimum-variance portfolios: pick the asset with the **lowest variance** in each cluster.
- Alternatively, practitioners can use Sharpe ratio, volatility, or other idiosyncratic metrics.

---

## 6. Implementation Notes & Edge Cases

1. **CORD computation:** Naive triple loop is $O(d^3)$. For $d = 500$, this is manageable in NumPy (~0.1-1s). We vectorize over $l$ to avoid Python loops.

2. **Whitening for tail estimation:** $\hat{\rho}^{-1/2}$ is computed via eigendecomposition. If $\hat{\rho}$ is near-singular, add a small jitter to eigenvalues before inversion.

3. **Singular correlations:** If two assets have perfectly identical return series, their sample correlation is 1 and standard deviation may be zero. We drop such assets or add small noise.

4. **Empty cluster count range:** If no $\varepsilon$ yields $K \in U$, the paper suggests relaxing $U$ or expanding the search range. In our implementation, we fall back to the $\varepsilon$ yielding the closest $K$ to the midpoint of $U$.

5. **Grid search efficiency:** PARTITION is called $n_g$ times. Each call is $O(d^2)$. With $d=500, n_g=100$, total runtime is a few seconds.

6. **Return calculation:** The paper uses daily returns $R_t = (P_t - P_{t-1}) / P_{t-1}$.

---

## 7. Benchmark Datasets

The paper uses:
- **S&P 500 constituents** from Compustat via WRDS, Jan 1996 -- Jan 2020.
- Daily closing prices.
- Lookback window: $n = 500$ trading days.
- Filtering: discard stocks with < 5 years history or > 5% missing data.
- Missing prices: linearly interpolated.

Our implementation uses the **S&P 500 data from the parent task** (5 years of daily OHLCV from Yahoo Finance, 2021-05-10 to 2026-05-06, 503 tickers).

---

## 8. Deviations from Paper

1. **Data source:** We use Yahoo Finance data instead of Compustat/WRDS. The time range is different (2021-2026 vs 1996-2020), but the methodology is identical.
2. **Pre-processing:** We compute daily returns from Close prices and drop tickers with any NaN in the lookback window, rather than interpolating. This is stricter but avoids introducing synthetic data.
3. **Missing values:** If $\hat{\rho}$ is not positive definite (can happen with identical return series), we add a small diagonal jitter ($10^{-6}$) before computing $\hat{\rho}^{-1/2}$.
4. **No portfolio construction backtest:** The task asks for clustering implementation only, not the full portfolio backtest with risk parity / min-variance / mean-variance allocation. We report cluster statistics and selected assets instead.

---

## 9. References

- Tang, W., Xu, X., & Zhou, X. Y. (2021). Asset Selection via Correlation Blockmodel Clustering. *arXiv preprint arXiv:2103.14506*.
- Bunea, F., et al. (2016). Model selection for high-dimensional regression with dependent observations.
- Markowitz, H. (1952). Portfolio Selection. *The Journal of Finance*.
