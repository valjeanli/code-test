# Regime Detection Evaluation Summary

## Comparison of All 6 Coder Implementations

**Evaluator**: software-tester (kanban task t_1066c8ca)
**Date**: 2026-05-09
**Paper**: arXiv:2306.15835 — Non-parametric online market regime detection (Horvath & Issa, 2023)
**Data**: S&P 500 OHLCV, evaluation window 2021-05-10 to 2026-05-06
**Spec**: T2 output_spec.json v1.0 + scoring_rubric.md

---

## Leaderboard

| Rank | Coder | Readme (10) | Code (10) | Output Compliance (10) | Output Quality (10) | T2 Spec Adherence (10) | **Total (50)** | Grade |
|------|-------|:-----------:|:---------:|:---------------------:|:-------------------:|:----------------------:|:--------------:|:-----:|
| **1** | **minimax** | 9 | 8 | 9 | 6 | 9 | **41** | B |
| **2** | **mimo** | 7 | 7 | 5 | 5 | 5 | **29** | D |
| **3** | **kimi** | 10 | 7 | 0 | 6 | 0 | **23** | D |
| **4** | **deepseek** | 9 | 9 | 0 | 0 | 2 | **20** | F |
| **5** | **glm** | 8 | 6 | 0 | 5 | 0 | **19** | F |
| **6** | **qwen** | 6 | 6 | 1 | 3 | 1 | **17** | F |

---

## Per-Coder Breakdown

### 1st place: minimax — 41/50 (B)

**Best at**: Output compliance and code quality

**Summary**: Minimax is the only coder that produced a fully spec-compliant output with all 7 CSV columns (4 required + 3 optional) and a complete metadata JSON. The readme is thorough with clear pseudocode and explicit deviations from the paper. The code is clean, self-contained (numpy + pandas only), and well-documented. However, the detection itself failed in the evaluation window — all 1,168 rows show the entire 2021-2026 period as a single "sideways" regime with zero transitions, missing the 2022 bear market and subsequent bull run.

**Key stats**:
- 10 total regime transitions across 20 years
- 0 transitions in the evaluation window (2021-2026)
- 3 regimes (bull, bear, sideways) but only sideways detected
- Confidence: static 1.0 for all rows

---

### 2nd place: mimo — 29/50 (D)

**Best at**: Technical implementation

**Summary**: Mimo produced the most technically complete implementation with proper truncated signature computation and two detection approaches (batch and online). The code includes auto-evaluator with weighted L-lag scoring. The `sig_mmd_online_regimes.csv` is well-formed with correct columns. However, the main CSV has `predicted_regime=-1` (violating spec), and the online detector outputs all rows as regime 0 with flat 0.5 confidence — detecting nothing.

**Key stats**:
- 5032 output rows (2006-2025)
- 2 CSV output files (one with -1 values)
- 3 regimes (bull/bear/sideways) but only "bull" detected
- 36 regime transitions detected in output text (but not reflected in CSV)

---

### 3rd place: kimi — 23/50 (D)

**Best at**: Paper understanding and mathematical rigor

**Summary**: Kimi has the most detailed readme (265 lines) with comprehensive mathematical exposition, proper algorithm pseudocode, and edge case documentation. It's the **only** implementation that uses a proper path signature library (`iisignature`). Successfully detected 5 major market events (1987 crash, 2008 GFC, 2020 COVID). However, no spec-compliant output files were produced — only a text file with MMD scores.

**Key stats**:
- 5 detected change points in 50+ years (too conservative)
- Uses global 98th percentile threshold
- 14,209 data points processed (1970-2026)
- Uses iisignature library (depth 4)
- Bootstrap with 300 replications

---

### 4th place: deepseek — 20/50 (F)

**Best at**: Code architecture and spec-aware design

**Summary**: Deepseek has the best code quality of all 6 — 891 lines, well-organized with proper output generation functions that strictly follow output_spec.json v1.0. The readme is also excellent with comprehensive paper understanding. However, the implementation **crashed during clustering** and produced no usable outputs. Additionally, the algorithm detected an absurd 78% of days as regime changes (3896/4988), indicating a fundamental bug in score normalization or threshold calculation.

**Key stats**:
- 3896/4988 regime changes detected (78% — clearly pathological)
- 8 maximum regimes (too many)
- Crashed during agglomerative clustering step
- Uses truncated signatures (level 2)

---

### 5th place: glm — 19/50 (F)

**Best at**: Mathematical depth

**Summary**: GLM's readme demonstrates the deepest mathematical understanding of the paper, discussing the signature kernel PDE, rank-2 MMD, and path transformations in detail. Detected 25 regime changes (2010-2020) with key events identified. However, the data range does not cover the evaluation window (2021-2026), no spec-compliant output files exist, and the code lacks output generation functions entirely.

**Key stats**:
- 25 detected regime changes (2010-2020)
- 3522 data points (wrong date range)
- No CSV or JSON outputs
- Best mathematical explanation of rank-2 signatures

---

### 6th place: qwen — 17/50 (F)

**Best at**: Novel approach

**Summary**: Qwen took a unique but non-conforming approach — analyzing 10 individual stocks instead of S&P 500. The CSV output exists but has severe violations: `predicted_regime` values start at 10 (not 0), only 57 rows (monthly sampling), and regime labels are non-standard. The multi-stock analysis produces only 6 ensemble points per stock, which is insufficient for meaningful regime detection.

**Key stats**:
- 10 stocks analyzed (not S&P 500)
- 57 total output rows (6 per stock)
- Regime labels start at 10 (violates spec)
- 11 total change points across all stocks

---

## Cross-Cutting Observations

### Output Spec Compliance Crisis

Only **1 out of 6** coders (minimax) produced properly formatted outputs matching output_spec.json. The rest either:
- Produced no CSV/JSON files at all (deepseek, kimi, glm)
- Produced CSVs with invalid values (mimo: -1 regimes, qwen: 10-indexed regimes)
- Produced outputs for wrong assets (qwen: individual stocks)

This is the single biggest failure across all implementations.

### Detection Quality Issues

| Coder | Transitions Detected | Evaluation Window | Quality Issue |
|-------|---------------------|-------------------|---------------|
| minimax | 10 (20yr) | All sideways | Zero transitions in eval window |
| glm | 25 (10yr) | Not covered | Wrong data range |
| mimo | 36 (20yr) | Regime 0 only | Zero actual regime changes |
| deepseek | 3896 (20yr) | N/A (crashed) | 78% rate = pathological |
| kimi | 5 (50yr) | N/A (no CSV) | Too conservative |
| qwen | 11 (10 stocks) | Monthly samples | Wrong asset, wrong sampling |

### Code Quality Ranking

1. **deepseek** (9/10) — 891 lines, modular, spec-aware, excellent structure
2. **minimax** (8/10) — 518 lines, clean, self-contained
3. **kimi** (7/10) — Proper library usage (iisignature), bootstrap
4. **mimo** (7/10) — Two implementations, proper signature computation
5. **qwen** (6/10) — Multi-stock support, functional
6. **glm** (6/10) — Functional but lacks output pipeline

### Paper Understanding Ranking

1. **deepseek** (9/10) — Most comprehensive readme
2. **kimi** (10/10) — Most detailed mathematical exposition
3. **glm** (8/10) — Deepest mathematical depth (PDE, rank-2)
4. **minimax** (9/10) — Clear and practical
5. **mimo** (7/10) — Good but less detailed
6. **qwen** (6/10) — Adequate coverage

---

## Recommendations

1. **For a production regime detection system**: Use minimax's output format (spec-compliant), combined with deepseek's algorithm design (most faithful to paper). The deepseek code just needs its clustering bug fixed and its threshold normalized.

2. **For output spec compliance**: Require automated validation — a simple script that checks `predicted_regime` range, row count, date range, and column presence would have caught all 5 non-compliant outputs before submission.

3. **For the evaluation framework**: The scoring rubric is excellent but should include a baseline compliance check that gates the rest of the evaluation. No output files = 0 total score.

4. **For future task design**: All coders should be required to run a validation script before marking their task complete. The output_spec.json has a `deliverable_checklist` — this should be automated.
