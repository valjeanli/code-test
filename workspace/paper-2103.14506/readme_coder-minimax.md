# Paper 2103.14506: Asset Selection via Correlation Blockmodel Clustering

**Authors:** Wenpin Tang, Xiao Xu, Xun Yu Zhou (Columbia University)
**Venue:** arXiv:2103.14506 (q-fin.PM), 46 pages, 9 figures, 8 tables
**Submitted:** March 2021, revised August 2021

---

## 1. Problem Statement

Given a universe of d financial assets (e.g., S&P 500 stocks), identify a small subset of stocks that approximates the diversification level of the entire universe. The goal is to cluster assets so that:
- **Criterion 1:** Assets within the same cluster are highly correlated with each other
- **Criterion 2:** Assets in different clusters have the same correlation with all other assets

The second criterion is novel and derived from portfolio theory: selecting one low-variance asset per cluster yields a well-diversified portfolio.

---

## 2. Correlation Blockmodel

### 2.1 Model Definition

Let $X_i$ be the standardized return of asset $i$, $i = 1, \dots, d$. The population correlation matrix $\rho = (\rho_{ij})$ follows a **K-group correlation blockmodel** if:

$$\rho_{ij} = \begin{cases}
\beta_k & \text{if } i, j \in G_k \text{ (within-cluster)} \\
\gamma_{k\ell} & \text{if } i \in G_k, j \in G_\ell, k \neq \ell \text{ (between-cluster)}
\end{cases}$$

where $\{G_1, \dots, G_K\}$ is a partition of $\{1, \dots, d\}$ and $\gamma_{k\ell} = \gamma_{\ell k}$.

### 2.2 Coarsest Partition Theorem (Theorem 1)

There exists a unique **coarsest partition** $G^*$ such that:
- Two assets $X_i$ and $X_j$ belong to different clusters in $G^*$ **iff** $\max_{l \neq i,j} |\rho_{il} - \rho_{jl}| > 0$

The coarsest partition is identifiable from the population correlation matrix.

### 2.3 Minimum Variance Portfolio Theorem (Theorem 2)

When the correlation blockmodel holds with coarsest partition $G^* = \{G_1, \dots, G_K\}$, the minimum variance portfolio by selecting one asset from each cluster is the asset with **lowest variance** in each cluster.

---

## 3. Sample Estimation

From $n$ observations of $d$-dimensional returns $X_1, \dots, X_n$, compute the **sample correlation matrix** $\hat{\rho} = (\hat{\rho}_{ij})$.

Define the **sample correlation difference** (CORD):

$$\widehat{\text{CORD}}_{ij} = \max_{l \neq i,j} |\hat{\rho}_{il} - \hat{\rho}_{jl}|$$

This estimates $\max_{l \neq i,j} |\rho_{il} - \rho_{jl}|$.

---

## 4. The PARTITION Procedure

**Input:** Dissimilarity matrix $\hat{D}(i,j) = \widehat{\text{CORD}}_{ij}$, threshold $\varepsilon > 0$

```
PROCEDURE PARTITION(D̂, ε):
  Active ← {1, ..., d}
  Clusters ← []

  while Active is not empty:
    Find (i*, j*) = argmin_{i,j in Active, i<j} D̂(i,j)

    if D̂(i*, j*) > ε:
      # Most similar pair is too dissimilar → singleton
      Clusters ← Clusters ∪ { {i*} }
      Active ← Active \ {i*}
    else:
      # Form new cluster around (i*, j*)
      NewCluster ← {i*, j*}
      Core ← {i*, j*}
      Active ← Active \ {i*, j*}

      while True:
        # Find all assets similar to current core
        Similar ← {l in Active : ∃ c in Core s.t. D̂(l,c) ≤ ε}
        if Similar is empty:
          break
        # Expand cluster
        NewCluster ← NewCluster ∪ Similar
        Core ← Similar
        Active ← Active \ Similar

      # Check cluster quality
      if |NewCluster| == 1:
        Clusters ← Clusters ∪ {NewCluster}
      else:
        Clusters ← Clusters ∪ {NewCluster}

  return Clusters
```

**Key properties:**
- Does NOT require number of clusters $K$ as input — determined by $\varepsilon$
- Generalizes the CORD algorithm (Bunea et al., 2016)
- Complexity: $O(d^2)$ per call

### 4.1 Statistical Guarantee (Theorem 3)

Under the correlation blockmodel with $n$ observations and $d$ assets, if:
$$\varepsilon \approx \sqrt{\log d / n}$$

and the cluster separation condition holds, then PARTITION recovers $G^*$ with high probability.

---

## 5. Tuning the Threshold $\varepsilon$

### 5.1 Intra-Cluster Correlation

For a given partition $\hat{G}_\varepsilon$ from threshold $\varepsilon$, define:

$$\hat{r}_\varepsilon = \frac{1}{\sum_{k=1}^{K} |G_k|} \sum_{k=1}^{K} \sum_{i,j \in G_k, i<j} \hat{\rho}_{ij}$$

### 5.2 Three Rules for Choosing $\varepsilon$

**Rule 1 (Existence):** $\varepsilon$ must satisfy $\hat{\tau} < \varepsilon < \hat{\Delta} - \hat{\tau}$ where $\hat{\tau}$ is the sampling error bound and $\hat{\Delta}$ is the minimum separation.

**Rule 2 (Intra-cluster correlation):** Among valid $\varepsilon$, choose the one maximizing $\hat{r}_\varepsilon$.

**Rule 3 (Cluster count constraint):** Restrict $K \in [K_{\min}, K_{\max}]$ (authors use $K \in [15, 25]$ for S&P 500).

### 5.3 Slope Method for $\hat{\tau}$

Plot $Y_i = \log(\text{inverse within-cluster CORD})$ vs $\log(L_i)$ where $L_i$ is window length. Fit linear regression — slope estimates $1/\alpha$, intersection estimates $\log L$.

---

## 6. The ACC Algorithm (Algorithm 1)

**Input:** Return matrix $X$ ($n \times d$), grid params $a, b, n_g$, cluster range $U = [K_{\min}, K_{\max}]$, tail param $k$, number of large obs

```
ALGORITHM ACC(X, a, b, n_g, U, k):

  # Step 1: Standardize
  X ← standardize columns of X to zero mean, unit variance

  # Step 2: Compute sample correlation
  Σ ← sample correlation matrix of X

  # Step 3: Compute CORD dissimilarity matrix
  for each pair (i,j):
    ĈORD(i,j) ← max_{l ≠ i,j} |Σ_il - Σ_jl|

  # Step 4: Estimate sampling error τ via slope method
  for each grid ε_i in [a, b] with n_g points:
    Run PARTITION(ĈORD, ε_i) → G_i
    Record K_i = |G_i|, r̄_i = intra-cluster correlation
  Filter: keep ε_i where K_i ∈ U
  Fit: log(Y) ~ α·log(L) + const on tail → get slope 1/α, intercept
  τ ← exp(intercept)

  # Step 5: Calibrate ε via intra-cluster correlation
  Candidate thresholds ← [ε : τ < ε < (min separation) - τ]
  Choose ε* ← argmax_{ε in candidates} r̄_ε

  # Step 6: Final partition
  return PARTITION(ĈORD, ε*)

OUTPUT: Cluster partition Ĝ
```

**Complexity:** $O(nd^2 + d^3)$ arithmetic operations.

---

## 7. Asset Selection Procedure

After obtaining clusters $\hat{G} = \{G_1, \dots, G_K\}$:

1. **Select one stock per cluster:** Choose the stock with the lowest variance in each cluster (per Theorem 2)
2. **Number of selected stocks:** Equals number of clusters $K$ (typically 15–25)
3. **Portfolio construction:** Equal-weight or minimum-variance among the selected stocks

---

## 8. Implementation Notes

### 8.1 Data Requirements
- **n = 504** trading days of daily returns (our implementation; paper uses 500)
- **d** assets (S&P 500 ≈ 500 assets)
- Clustering rebalanced monthly (authors rebalance annually for portfolios)

### 8.2 Key Parameters Used in This Implementation
| Parameter | Value | Meaning |
|-----------|-------|---------|
| $n$ | 504 | Lookback window (trading days) |
| $k$ | $n/4 = 126$ | For slope estimation tail |
| $a, b$ | grid range | For $\varepsilon$ search |
| $n_g$ | 40 | Resolution of $\varepsilon$ grid |
| $U$ | $[10, 35]$ | Range of number of clusters |

### 8.3 Deviations from Paper
- **ε-tuning**: We use an empirical CORD percentile approach instead of the slope method. We scan CORD values at coarse percentiles to find the eps range bracketing the target cluster count, then fine-grid search within that range.
- **Rule 3 only**: Our grid search primarily enforces Rule 3 (cluster count range $[10, 35]$). Rule 2 (max intra-cluster correlation) selects among ε values that produce valid cluster counts.
- **τ estimation**: The theoretical τ = √(log d / n) is replaced by empirical CORD distribution analysis for robustness on real data.
- **Data**: We use 504-day (≈ 2 trading years) rolling windows of S&P 500 log returns. The paper uses 500-day windows.
- We use the sample correlation matrix directly without the tail-parameter adjustment described in Section 2.4

### 8.4 ASSUMPTION 1 (for Theorem 3)
The return vectors $X_i$ are i.i.d. with sub-Gaussian tails. This is the standard assumption for concentration inequalities used in the proof.

---

## 9. Output Files

- `readme_coder-minimax.md` — This document
- `code_coder-minimax.py` — Full Python implementation
- `output_coder-minimax.txt` — Run output with clustering results

---

## 10. References

- Bunea, Giraud, Luo (2016) — CORD algorithm (predecessor to PARTITION)
- Markowitz (1952, 1959) — Mean-variance portfolio theory
- Mantegna (1999) — Hierarchical clustering in financial markets
- Kaufman & Rousseeuw (1990) — k-medoids clustering
