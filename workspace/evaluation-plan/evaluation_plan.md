# Regime Detection Evaluation Plan
## S&P 500 Stock Movement — Multi-Algorithm Benchmark

**Version**: 1.0
**Date**: 2026-05-09
**Author**: coder-deepseek (kanban task t_d9d3cdc6)

---

## 1. Problem Statement

Regime detection (market state classification) is inherently **unsupervised** — there is no ground-truth label for "what regime are we in?" This makes evaluation fundamentally different from supervised classification (where you have accuracy, precision, recall against known labels).

We need an evaluation framework that:
- **Does not require labeled ground truth** (algorithms are unsupervised)
- **Rewards regimes that are statistically real** (distinct return/vol profiles)
- **Rewards regimes that are economically meaningful** (align with known stress/calm periods)
- **Rewards regimes that are actionable** (improve trading outcomes in walk-forward)
- **Rewards stability** (regimes shouldn't flip every day)
- **Is fair across algorithm families** (HMM, clustering, change-point detection, ML-based)

---

## 2. Proxy Ground Truths

While we don't have true labels, we can construct **proxy ground truths** from market data to evaluate economic alignment:

| Proxy | Definition | Purpose |
|-------|-----------|---------|
| **VIQ** (VIX Quartiles) | Assign each day to Q1-Q4 based on VIX level over full sample | Does the algorithm separate high-vol from low-vol periods? |
| **DD3** (Drawdown Terciles) | Assign each day to D1-D3 based on SPX drawdown from 252-day high | Does the algorithm detect bear markets? |
| **Crisis Flags** | Binary: 1 during known stress (COVID crash Mar 2020, 2022 bear, 2008 GFC if in sample) | Does the algorithm flag extreme events? |
| **Trend Flags** | Binary: 1 when SPX 50-day MA > 200-day MA | Does the algorithm separate bull from bear trends? |

These are used **only for the Economic Alignment metric** (Metric 3). They are NOT ground truth — they are sanity checks that regimes capture real economic variation.

---

## 3. Five Evaluation Metrics

### Metric 1: Regime Separability (25 points)

**What it measures**: Are the regimes statistically distinct? Within each regime, returns and volatility should be similar. Across regimes, they should be different.

**Method**: 
- For each regime k, compute the regime-conditional distribution of S&P 500 daily returns: mean μₖ, std σₖ
- Compute the **silhouette score** using the regime labels on a feature vector of [20-day rolling return, 20-day rolling volatility]
- Normalize to [0,1] range (silhouette ∈ [-1,1], clip negative to 0)
- Alternative if silhouette is degenerate (e.g., only 1 regime): use **ANOVA F-statistic** on regime-conditional returns

**Scoring function**:
```
separability_score = max(0, silhouette_score) * 25
```
If silhouette > 0.5 → 25 (excellent separation)
If silhouette ≈ 0.2 → 10 (moderate separation)
If silhouette ≤ 0 → 0 (no separation, worse than random)

**Rationale**: If an algorithm claims to find "bull" and "bear" regimes but both have identical return distributions, the regimes aren't real. This is the most fundamental validity check.

---

### Metric 2: Regime Stability (15 points)

**What it measures**: Good regimes are persistent. A regime that changes every 2-3 days is noise, not a meaningful market state.

**Method**:
- Compute the **average dwell time**: mean number of consecutive trading days spent in each regime before a transition
- Compare to expected dwell time under random switching with same transition probability
- Also compute **transition frequency**: fraction of days where regime changes

**Scoring function**:
```
avg_dwell_days = mean(dwell_times)
ideal_dwell = 21  # ~1 month minimum for a "meaningful" regime

if avg_dwell_days >= ideal_dwell:
    stability_score = 15
elif avg_dwell_days >= 10:
    stability_score = 10 + 5 * (avg_dwell_days - 10) / 11
elif avg_dwell_days >= 5:
    stability_score = 5 + 5 * (avg_dwell_days - 5) / 5
else:
    stability_score = avg_dwell_days / 5 * 5  # linear below 5 days
```

**Edge cases**:
- Single-regime algorithms (no transitions): score = 10 (penalized for not detecting changes, but not zero)
- Extremely frequent switching (>30% of days): score = 0

**Rationale**: Market regimes last weeks to months. Daily flipping is noise. But we don't want to over-reward algorithms that never change regime either, hence the cap.

---

### Metric 3: Economic Alignment (25 points)

**What it measures**: Do the detected regimes align with economically meaningful market states (high VIX, deep drawdowns, trend direction)?

**Method**: 
- Compute **Adjusted Rand Index (ARI)** between predicted regimes and each proxy ground truth (VIQ, DD3, Crisis, Trend)
- Take the **maximum ARI** across all proxies (an algorithm might specialize in detecting volatility regimes OR trend regimes — both are valid)
- ARI ∈ [-1, 1]; clip negative to 0

**Scoring function**:
```
ari_max = max(ARI(regimes, VIQ), ARI(regimes, DD3), ARI(regimes, Crisis), ARI(regimes, Trend))
alignment_score = ari_max * 25
```

**Why ARI and not mutual information?** ARI corrects for chance agreement (random labelings get ARI ≈ 0). It handles different numbers of clusters in predicted vs. proxy regimes.

**Why max across proxies?** Different algorithms may capture different aspects of market structure. A mean-reversion detector may align with VIX; a trend-follower may align with Trend. Both are valid. Taking max rewards specialization.

**Rationale**: While regimes are unsupervised, they should capture real economic variation. If predicted regimes have zero correlation with VIX, drawdowns, or trend — what are they capturing?

---

### Metric 4: Sharpe Ratio Improvement (25 points)

**What it measures**: Does using regime information improve risk-adjusted returns in a walk-forward setting?

**Method (Walk-Forward to avoid look-ahead bias)**:
1. Use an expanding window: first 252 trading days (1 year) as initial training
2. Train regime detection on window [0, t-1]
3. Predict regime for day t (strictly out-of-sample)
4. Apply a simple regime-conditional allocation rule:
   - If the most recent regime had positive mean return: allocate 100% to SPY
   - If the most recent regime had negative mean return: allocate 100% to cash (or SHY / BIL)
5. Advance window by 1 day, retrain monthly (every 21 trading days)
6. Compare Sharpe ratio of this strategy vs. buy-and-hold SPY

**Scoring function**:
```
sharpe_strategy = regime_adaptive_sharpe
sharpe_benchmark = buy_and_hold_sharpe
sharpe_diff = sharpe_strategy - sharpe_benchmark

if sharpe_diff >= 0.5:
    sharpe_score = 25     # massive improvement
elif sharpe_diff >= 0.2:
    sharpe_score = 15 + 10 * (sharpe_diff - 0.2) / 0.3
elif sharpe_diff >= 0.0:
    sharpe_score = 5 + 10 * sharpe_diff / 0.2
elif sharpe_diff >= -0.1:
    sharpe_score = 5 * (1 + sharpe_diff / 0.1)  # mild underperformance
else:
    sharpe_score = 0      # significant underperformance
```

**Edge cases**:
- If strategy Sharpe is negative but buy-and-hold is also negative (bear market period): score = max(5, ...) floor
- Single-regime algorithms: effectively buy-and-hold, expect score ≈ 5 (neutral)

**Rationale**: This is the acid test. If knowing the regime doesn't help you make better allocation decisions, what's the point? The walk-forward design prevents data snooping.

---

### Metric 5: Transition Responsiveness (10 points)

**What it measures**: When a major market regime shift occurs (defined by VIX spike > 2σ above 20-day MA), how quickly does the algorithm register a regime change?

**Method**:
- Identify VIX spike events: days where VIX > mean(VIX) + 2 * std(VIX) over trailing 252 days
- For each spike event, find the nearest regime transition in the algorithm's output within a [-5, +10] day window
- Measure the **lag**: days from VIX spike to detected transition (positive = late, negative = early, may be anticipatory)

**Scoring function**:
```
For each spike event with a detected transition within window:
    lag_days = |transition_date - spike_date|, capped at 10
    event_score += (10 - lag_days) / 10

Total events: N_spikes
responsiveness_score = 10 * (sum(event_scores) / N_spikes)
```

If no VIX spike events in the test period (unusually calm market): score = 10 (default, no penalty).

**Rationale**: A good regime detection system should respond to genuine market dislocations. This is a minor metric (10%) because it only tests extreme events, not everyday regime quality.

---

## 4. Composite Score

| Metric | Weight | Max Points |
|--------|--------|------------|
| Regime Separability | 25% | 25 |
| Regime Stability | 15% | 15 |
| Economic Alignment | 25% | 25 |
| Sharpe Improvement | 25% | 25 |
| Transition Responsiveness | 10% | 10 |
| **Total** | **100%** | **100** |

---

## 5. Data Requirements

All evaluations use the S&P 500 OHLCV dataset at `workspace/sp500_data/sp500_ohlcv_5yr.parquet` (produced by task t_dc995cda).

Additional data needed:
- **VIX index** (^VIX from yfinance) — for proxy ground truths and transition detection
- **SPY / SPX** (S&P 500 ETF or index) — for benchmark comparison
- **SHY or BIL** (short-term Treasury ETF) — for cash proxy in Sharpe calculation
- **Risk-free rate** — 3-month T-bill rate (or use 0% for simplicity)

Date range: 2021-05-10 to 2026-05-06 (matching the 5-year dataset).

---

## 6. Fairness Across Algorithm Types

| Algorithm Type | Strengths | Potential Weaknesses on These Metrics |
|----------------|-----------|---------------------------------------|
| HMM (Hidden Markov Model) | Natural probabilistic transitions, good dwell times | May need pre-specified K (number of regimes) |
| Clustering (K-means, ACC, spectral) | Good separability by design | Often unstable (frequent switching), poor stability |
| Change-point detection (PELT, Binary Segmentation) | Excellent transition detection | May produce too many or too few regimes |
| ML-based (Isolation Forest, autoencoders) | Good at anomaly/regime detection | Hard to calibrate; may overfit |
| Trend-following (MA crossover, momentum) | Naturally stable, good alignment with Trend proxy | May miss volatility regimes |

The scoring system is designed to not favor any single approach. Each algorithm can excel on different metrics.

---

## 7. References

1. Ang, A., & Timmermann, A. (2012). "Regime Changes and Financial Markets." *Annual Review of Financial Economics*.
2. Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*.
3. Kritzman, M., Page, S., & Turkington, D. (2012). "Regime Shifts: Implications for Dynamic Strategies." *Financial Analysts Journal*.
4. Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics*.
5. Hubert, L., & Arabie, P. (1985). "Comparing Partitions." *Journal of Classification*. (Adjusted Rand Index)
