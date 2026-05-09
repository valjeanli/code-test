# Regime Detection Scoring Rubric
## 0-100 Weighted Composite Score

**Version**: 1.0
**Date**: 2026-05-09
**Evaluator**: Task #4 implementation
**Input**: Output files from Task #3 conforming to `output_spec.json`

---

## Scoring Summary

| # | Metric | Weight | Max Points | Category |
|---|--------|--------|------------|----------|
| 1 | Regime Separability | 25% | 25 | Statistical validity |
| 2 | Regime Stability | 15% | 15 | Practical usability |
| 3 | Economic Alignment | 25% | 25 | Economic meaningfulness |
| 4 | Sharpe Improvement | 25% | 25 | Trading efficacy |
| 5 | Transition Responsiveness | 10% | 10 | Event detection |
| **Total** | | **100%** | **100** | |

---

## Metric 1: Regime Separability (0-25 points)

### Definition
Measures how statistically distinct the regime-conditional return distributions are. Within-regime returns should cluster tightly; between-regime returns should separate.

### Input Data
- `sp500_ohlcv_5yr.parquet`: Use SPY or index-level daily returns
- T3 output: `predicted_regime` column

### Calculation

**Step 1: Build feature matrix**
For each day `t`, construct a 2D feature vector:
```
features[t] = [rolling_return_20d[t], rolling_vol_20d[t]]
```
where:
- `rolling_return_20d` = cumulative log return over prior 20 trading days
- `rolling_vol_20d` = annualized standard deviation of daily returns over prior 20 days

**Step 2: Compute silhouette score**
```
from sklearn.metrics import silhouette_score
sil = silhouette_score(features, predicted_regime_labels)
```
Using Euclidean distance. `sil ∈ [-1, 1]`.

**Step 3: Score mapping**
```
if sil >= 0.50:    score = 25.0
elif sil >= 0.30:  score = 20.0 + (sil - 0.30) / 0.20 * 5.0   # linear 20-25
elif sil >= 0.15:  score = 10.0 + (sil - 0.15) / 0.15 * 10.0  # linear 10-20
elif sil >= 0.05:  score = 5.0  + (sil - 0.05) / 0.10 * 5.0   # linear 5-10
elif sil >  0.00:  score = sil / 0.05 * 5.0                    # linear 0-5
else:             score = 0.0
```

### Edge Cases
| Case | Handling |
|------|----------|
| Single regime (num_regimes == 1) | silhouette_score raises ValueError. Fall back: score = 0 (no separation possible) |
| NaN values in features | Forward-fill then backward-fill. If still NaN, exclude those rows |
| All identical predictions | Silhouette → 0, score = 0 |
| Negative silhouette | Clip to 0 |

### Interpretation Guide
- **20-25**: Excellent — regimes have clearly different return/volatility profiles
- **10-19**: Good — regimes are distinguishable but overlap exists
- **5-9**: Marginal — regimes are weakly separated
- **0-4**: Poor — regimes are not statistically distinct from random partitioning

---

## Metric 2: Regime Stability (0-15 points)

### Definition
Measures how persistent regimes are. Meaningful market regimes last weeks to months, not days.

### Input Data
- T3 output: `predicted_regime` and `transition_flag` columns

### Calculation

**Step 1: Compute dwell times**
For each regime segment (consecutive days with same `predicted_regime`), count the number of trading days. This is the dwell time.

```
def compute_dwell_times(labels):
    dwells = []
    current = labels[0]
    count = 1
    for i in range(1, len(labels)):
        if labels[i] == current:
            count += 1
        else:
            dwells.append(count)
            current = labels[i]
            count = 1
    dwells.append(count)
    return dwells
```

**Step 2: Score**
```
avg_dwell = mean(dwell_times)
T = len(labels)
transition_rate = sum(transition_flags) / (T - 1)  # fraction of days with transition

if avg_dwell >= 21:           score = 15.0
elif avg_dwell >= 10:         score = 10.0 + (avg_dwell - 10) / 11 * 5.0
elif avg_dwell >= 5:          score = 5.0  + (avg_dwell - 5)  / 5  * 5.0
elif avg_dwell >= 2:          score = avg_dwell / 5 * 5.0
else:                         score = 0.0
```

**Step 3: Apply transition penalty**
```
if transition_rate > 0.30:    score = 0.0   # flicker penalty (>30% of days switch)
elif transition_rate > 0.20:  score *= 0.5   # heavy penalty
elif transition_rate > 0.10:  score *= 0.8   # mild penalty
```

### Edge Cases
| Case | Handling |
|------|----------|
| num_regimes == 1 | avg_dwell = T (entire period). Score = 12 (stable but uninformative; partial credit) |
| Very short evaluation window (< 60 days) | Scale ideal_dwell proportionally: `ideal = max(5, T/12)` |
| All dwell_times = 1 | Score = 0 (regime changes every day) |

### Interpretation Guide
- **12-15**: Excellent — regimes persist for weeks/months
- **8-11**: Good — regimes are reasonably stable
- **4-7**: Marginal — regimes are short-lived
- **0-3**: Poor — excessive regime switching, likely noise

---

## Metric 3: Economic Alignment (0-25 points)

### Definition
Measures whether detected regimes align with economically meaningful market states (high/low VIX, deep/shallow drawdowns, bull/bear trends, crisis periods).

### Input Data
- yfinance: ^VIX daily close (downloaded by evaluator)
- `sp500_ohlcv_5yr.parquet`: SPY close for drawdown calculation
- T3 output: `predicted_regime` column

### Proxy Ground Truth Construction

**Proxy 1: VIQ (VIX Quartiles)** — 4 labels
```python
vix = fetch_vix(start_date, end_date)
vix['vix_quartile'] = pd.qcut(vix['close'], q=4, labels=[0, 1, 2, 3])
```

**Proxy 2: DD3 (Drawdown Terciles)** — 3 labels
```python
spy = load_spy_data()
rolling_high = spy['close'].rolling(252).max()
drawdown = spy['close'] / rolling_high - 1  # negative values
spy['dd_tercile'] = pd.qcut(drawdown, q=3, labels=[0, 1, 2])
```

**Proxy 3: CRISIS (Binary)** — 2 labels
```python
# VIX > 30 qualifies as "crisis" regime
vix['crisis'] = (vix['close'] > 30).astype(int)
```
If fewer than 5 crisis days in the evaluation window, this proxy is excluded.

**Proxy 4: TREND (Binary)** — 2 labels
```python
spy['sma_50'] = spy['close'].rolling(50).mean()
spy['sma_200'] = spy['close'].rolling(200).mean()
spy['trend'] = (spy['sma_50'] > spy['sma_200']).astype(int)
```

### Calculation

```python
from sklearn.metrics import adjusted_rand_score

proxies = {
    'VIQ': vix_quartile_labels,
    'DD3': drawdown_tercile_labels,
    'CRISIS': crisis_labels,   # excluded if <5 crisis days
    'TREND': trend_labels
}

ari_scores = {}
for name, proxy_labels in proxies.items():
    # Align dates, drop any NaN
    common_idx = predicted_regime.index.intersection(proxy_labels.dropna().index)
    ari_scores[name] = adjusted_rand_score(
        predicted_regime.loc[common_idx],
        proxy_labels.loc[common_idx]
    )

ari_max = max(ari_scores.values())  # max ARI across all valid proxies
alignment_score = max(0, ari_max) * 25.0  # clip negative to 0
```

### Edge Cases
| Case | Handling |
|------|----------|
| Single-regime algorithm | ARI = 0 with all proxies (no variation to match). Score = 0 |
| Missing VIX data | Exclude that proxy; use remaining ones |
| <5 crisis days | Exclude CRISIS proxy |
| num_regimes very high (>10) | ARI may be artificially low. Add note in evaluation report but don't adjust score |

### Interpretation Guide
- **20-25**: Excellent — regimes strongly align with real market state variation
- **10-19**: Good — regimes capture some economic structure
- **5-9**: Marginal — weak alignment with economic reality
- **0-4**: Poor — regimes don't correspond to any known market states

---

## Metric 4: Sharpe Ratio Improvement (0-25 points)

### Definition
Tests whether regime-aware tactical allocation outperforms buy-and-hold in a strict walk-forward setting (no look-ahead bias).

### Input Data
- `sp500_ohlcv_5yr.parquet`: SPY daily close
- yfinance: SHY (iShares 1-3 Year Treasury Bond ETF) daily close for cash proxy
- T3 output: `predicted_regime` column

### Calculation

**Step 1: Walk-forward setup**
```
initial_training_days = 252  # 1 year
retrain_frequency = 21       # re-fit every 21 trading days (~monthly)
risk_free_rate = 0.0         # annualized, use 0 for simplicity (or fetch 3M T-bill)
```

**Step 2: Per-regime expected return estimation**
At each retrain step t (every 21 days from day 252 onward):
```
# Use only data available UP TO day t-1 (strictly out-of-sample)
training_regimes = predicted_regime[0:t]
training_returns = spy_daily_return[0:t]

for each regime k:
    regime_returns = training_returns[training_regimes == k]
    regime_mean_return[k] = mean(regime_returns)
```

**Step 3: Tactical allocation rule**
For day t:
```
predicted_regime_t = predicted_regime[t]

if regime_mean_return[predicted_regime_t] > 0:
    allocation[t] = 100% SPY
else:
    allocation[t] = 100% SHY (cash proxy)
```

**Step 4: Compute Sharpe ratios**
```python
strategy_daily_returns = spy_daily_return * allocation_spy_pct + shy_daily_return * (1 - allocation_spy_pct)
benchmark_daily_returns = spy_daily_return  # 100% buy-and-hold

sharpe_strategy = (mean(strategy_daily_returns) * 252) / (std(strategy_daily_returns) * sqrt(252))
sharpe_benchmark = (mean(benchmark_daily_returns) * 252) / (std(benchmark_daily_returns) * sqrt(252))

sharpe_diff = sharpe_strategy - sharpe_benchmark
```

**Step 5: Score mapping**
```
if sharpe_diff >= 0.50:       score = 25.0
elif sharpe_diff >= 0.20:     score = 15.0 + (sharpe_diff - 0.20) / 0.30 * 10.0
elif sharpe_diff >= 0.00:     score = 5.0  + (sharpe_diff - 0.00) / 0.20 * 10.0
elif sharpe_diff >= -0.10:    score = 5.0  * (1.0 + sharpe_diff / 0.10)
else:                         score = 0.0
```

### Edge Cases
| Case | Handling |
|------|----------|
| num_regimes == 1 | Effectively buy-and-hold. Score = 5 (neutral baseline) |
| SHY data missing | Use 0% return for cash (slightly pessimistic but fair) |
| Strategy has zero trades (always cash) | Sharpe ≈ 0. Compare vs benchmark normally |
| Evaluation window < 252 days | Skip this metric entirely; weight redistributed proportionally to other metrics |
| Regime never appears in training | Default to cash allocation (conservative) |

### Interpretation Guide
- **20-25**: Exceptional — regime timing dramatically improves risk-adjusted returns
- **10-19**: Strong — regime allocation provides meaningful edge
- **5-9**: Neutral — comparable to buy-and-hold
- **0-4**: Harmful — regime-based allocation underperforms buy-and-hold

---

## Metric 5: Transition Responsiveness (0-10 points)

### Definition
Measures how quickly the algorithm detects bona fide market dislocations (VIX spikes > 2σ).

### Input Data
- yfinance: ^VIX daily close
- T3 output: `transition_flag` and `predicted_regime` columns

### Calculation

**Step 1: Identify VIX spike events**
```
vix_rolling_mean = vix['close'].rolling(252).mean()
vix_rolling_std = vix['close'].rolling(252).std()
spike_threshold = vix_rolling_mean + 2 * vix_rolling_std
spike_dates = vix.index[vix['close'] > spike_threshold]
```

**Step 2: Merge consecutive spike days**
Cluster spike days within 5 trading days of each other into single "events." Use the first day of each cluster as the event date.

**Step 3: Score each event**
```python
event_scores = []
search_window = 10  # trading days

for event_date in spike_event_dates:
    # Find nearest regime transition within [-5, +10] window
    window_start = event_date - timedelta(days=5)
    window_end = event_date + timedelta(days=10)
    
    transitions = t3_output[
        (t3_output['date'] >= window_start) &
        (t3_output['date'] <= window_end) &
        (t3_output['transition_flag'] == 1)
    ]
    
    if len(transitions) == 0:
        event_scores.append(0.0)  # missed entirely
    else:
        nearest = min(transitions['date'], key=lambda d: abs(d - event_date))
        lag_days = abs((nearest - event_date).days)
        event_scores.append(max(0, (10 - min(lag_days, 10)) / 10))
```

**Step 4: Aggregate**
```
if len(spike_event_dates) == 0:
    responsiveness_score = 10.0  # no events to test, full credit
else:
    responsiveness_score = 10.0 * mean(event_scores)
```

### Edge Cases
| Case | Handling |
|------|----------|
| Zero VIX spike events | Score = 10 (default, no penalty for calm markets) |
| num_regimes == 1 | No transitions possible → all event_scores = 0 → score = 0 |
| Transition flag always 0 | Score = 0 (algorithm claims no transitions happened) |
| Transition flag always 1 | Every day is a "transition." Find the day where regime actually changes; only count those |

### Interpretation Guide
- **8-10**: Excellent — detects market dislocations within 1-2 days
- **5-7**: Good — detects events but with some delay
- **2-4**: Marginal — misses many events or responds slowly
- **0-1**: Poor — does not respond to major market events

---

## Final Composite Score

```python
total_score = (
    separability_score +
    stability_score +
    alignment_score +
    sharpe_score +
    responsiveness_score
)
# total_score ∈ [0, 100]
```

### Grade Scale

| Score Range | Grade | Interpretation |
|-------------|-------|----------------|
| 85-100 | A | Exceptional regime detection — all metrics strong |
| 70-84 | B | Good — useful regimes with minor weaknesses |
| 55-69 | C | Adequate — some regimes visible but noisy or poorly timed |
| 40-54 | D | Weak — regimes barely distinguishable from random |
| 0-39 | F | Failed — regimes not useful for any practical purpose |

---

## Worked Example

### Hypothetical Algorithm: 2-State Gaussian HMM
- num_regimes = 2 (bull/bear)
- 1258 trading days evaluated

**Metric 1 — Separability**
- Silhouette score = 0.42
- Score = 20 + (0.42 - 0.30)/0.20 * 5 = **23.0**

**Metric 2 — Stability**
- Dwell times: [45, 12, 67, 8, 34, ...], avg = 18.3 days
- Transition rate = 0.08 (8% of days)
- Score = 10 + (18.3 - 10)/11 * 5 = **13.8**
- No transition penalty (0.08 < 0.10)

**Metric 3 — Economic Alignment**
- ARI vs VIQ = 0.45
- ARI vs DD3 = 0.38
- ARI vs CRISIS = 0.52
- ARI vs TREND = 0.61 ← max
- Score = 0.61 * 25 = **15.3**

**Metric 4 — Sharpe Improvement**
- Strategy Sharpe = 0.85
- Benchmark Sharpe = 0.62
- Diff = 0.23
- Score = 15 + (0.23 - 0.20)/0.30 * 10 = **16.0**

**Metric 5 — Transition Responsiveness**
- 4 VIX spike events, avg event score = 0.75
- Score = 10 * 0.75 = **7.5**

**Total**: 23.0 + 13.8 + 15.3 + 16.0 + 7.5 = **75.6 → Grade B**

Strengths: Strong separability, good stability
Weaknesses: Moderate alignment with economic proxies, could respond faster to shocks

---

## Implementation Notes for Evaluator (Task #4)

1. **All calculations must be deterministic** — seed any randomness (e.g., when using qcut for bins, the bins are based on the full sample so they are deterministic)
2. **Date alignment is critical** — all datasets (SPY, VIX, T3 output) must be aligned on trading dates before computing metrics
3. **Log all intermediate values** — the evaluator should output not just the final score but also each sub-score and its inputs (silhouette value, ARI values, dwell times, Sharpe ratios) for transparency
4. **Handle missing data gracefully** — if VIX or SHY data is unavailable for parts of the window, exclude those dates rather than failing
5. **Output format** — evaluator should produce a JSON evaluation report matching `evaluation_report_schema.json` (to be defined by evaluator task)
