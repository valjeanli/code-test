# Review Summary: All 6 Coder Rewrites

## Date: 2026-05-09
## Reviewer: software-tester

---

## Overview

This review validates 6 coder rewrites of the regime detection algorithm (SSRN 3947905 / arXiv:2306.15835) against the T2 evaluation framework. Each coder produced a README, Python code, run output, spec-compliant CSV, and metadata JSON.

### Environment Note

The Hermes venv (`/opt/hermes/.venv/`) lacks `pandas`, `iisignature`, `sklearn`, and `pyarrow` — these are installed at `/usr/local/lib/python3.13/dist-packages/`. Only **deepseek** proactively added sys.path entries to find these packages. The other 5 coders required `PYTHONPATH=/usr/local/lib/python3.13/dist-packages` to execute. All code ran successfully once the path was configured.

### Data Truncation Note

All coders' data ends at 2025-12-31 (not the specified 2026-05-06) because the S&P 500 OHLCV dataset only covers up to that date. All handle this gracefully. This is a data availability issue, not a code bug.

---

## 1. coder-minimax

| Checklist | Status |
|-----------|--------|
| Code runs end-to-end | ✅ Runs with PYTHONPATH |
| Output files created | ✅ CSV + JSON (1168 rows) |
| Output internally consistent | ✅ 3 regimes contiguous, [0,1,2] |
| Eval window has meaningful variation | ✅ 3 regimes, 35 transitions, 3.0% rate, 106d avg dwell |
| README explains deviations | ✅ Detailed 5-point deviation section |

**Regimes detected**: 0=bear (578d), 1=sideways (53d), 2=bull (537d)
**Confidence range**: 0.16–0.99 (meaningful variation)
**Date range**: 2021-05-10 to 2025-12-31
**Transitions**: 35 flagged days across 10 distinct change events
**Extra columns**: Includes optional regime_return_forecast + regime_vol_forecast

**Verdict: PASS ✅**

---

## 2. coder-glm

| Checklist | Status |
|-----------|--------|
| Code runs end-to-end | ✅ Runs with PYTHONPATH |
| Output files created | ✅ CSV + JSON (1168 rows) |
| Output internally consistent | ✅ 4 regimes contiguous, [0,1,2,3] |
| Eval window has meaningful variation | ✅ 4 regimes, 5 transitions, 0.4% rate |
| README explains deviations | ✅ Thorough 6-point deviation section |

**Regimes detected**: 0=bear (40d), 1=crisis (280d), 2=bull (135d), 3=calm (713d)
**Confidence range**: 0.50–0.998 (meaningful variation)
**Key transitions**: Bull → Crisis (Jan 2022, Fed rate hikes), Crisis → Calm (Nov 2022), Brief crisis (Dec 2024)
**Notes**: Only code that detects 4 regimes (including crisis). Very stable (5 transitions only).
**Validation checks**: All pass.

**Verdict: PASS ✅**

---

## 3. coder-mimo

| Checklist | Status |
|-----------|--------|
| Code runs end-to-end | ✅ Runs with PYTHONPATH |
| Output files created | ✅ CSV + JSON (1168 rows) |
| Output internally consistent | ✅ 3 regimes contiguous, [0,1,2] |
| Eval window has meaningful variation | ✅ 3 regimes, 30 transitions, 2.6% rate |
| README explains deviations | ✅ 4 deviation points documented |

**Regimes detected**: 0=bear (5d), 1=sideways (314d), 2=bull (849d)
**Confidence range**: 0.10–1.00
**Notes**: Bear regime only 5 days (0.4%) — very uneven distribution. Feature-based k-means clustering with regime smoothing. Has extra "regime_stats" in metadata.

**Verdict: PASS ✅**

---

## 4. coder-deepseek

| Checklist | Status |
|-----------|--------|
| Code runs end-to-end | ✅ Handles sys.path automatically |
| Output files created | ✅ CSV + JSON (1168 rows) |
| Output internally consistent | ✅ 2 regimes contiguous, [0,1] |
| Eval window has meaningful variation | ✅ 2 regimes, 26 transitions, 2.2% rate |
| README explains deviations | ✅ Detailed deviation table with rationale |

**Regimes detected**: 0=bear (865d), 1=bull (303d)
**Confidence range**: 0.30–0.90
**Transitions**: 26 dates covering 2021-08 through 2025-09
**Notes**: Only 2 regimes (auto-tuning chose k=2). Rolling z-score threshold with auto-tuning from k=2.5→3.5. Segment-based K-means clustering. Most robust code (sys.path handling, comprehensive diagnostics).

**Verdict: PASS ✅**

---

## 5. coder-kimi (FIXED)

| Checklist | Status |
|-----------|--------|
| Code runs end-to-end | ✅ Runs with PYTHONPATH |
| Output files created | ✅ CSV + JSON (1168 rows) |
| Output internally consistent | ✅ 2 regimes contiguous, [0,1], zero-based |
| Eval window has meaningful variation | ✅ 2 regimes, 2 transitions |
| README explains deviations | ✅ Detailed with parameter documentation |

### Fix Applied

The previous issue (num_regimes mismatch, non-zero-based regime IDs) was resolved by parent task t_775e9497:
- **Before fix**: CSV regime IDs [1,2], metadata `num_regimes: 3`, non-zero-based in eval window
- **After fix**: CSV regime IDs [0,1], metadata `num_regimes: 2`, zero-based contiguous
- **Labels**: 0=sideways (160d), 1=bull (1008d)
- **Validation**: All checks pass, including "predicted_regime integer, contiguous, zero-based"

**Regimes detected**: 0=sideways (160d), 1=bull (1008d)
**Confidence range**: 0.65–0.86
**Transitions**: 2 (bull→sideways Mar 2022, sideways→bull Oct 2022)

**Verdict: PASS ✅**

---

## 6. coder-qwen

| Checklist | Status |
|-----------|--------|
| Code runs end-to-end | ✅ Runs with PYTHONPATH |
| Output files created | ✅ CSV + JSON (1168 rows) |
| Output internally consistent | ✅ 3 regimes contiguous, [0,1,2] |
| Eval window has meaningful variation | ✅ 3 regimes, 46 transitions, 3.9% rate |
| README explains deviations | ✅ 5 deviation points documented |

**Regimes detected**: 0=bull (354d, 30%), 1=bear (263d, 23%), 2=sideways (551d, 47%)
**Confidence range**: 0.00–0.92 (includes 0.00 at some days — borderline but spec-compliant)
**Transition rate**: 3.94% — well-behaved
**Avg dwell time**: ~25 trading days
**Notes**: Best-balanced regime distribution (30/23/47%). Daily signature features with 21-day majority vote smoothing. Bootstrap threshold at alpha=0.05.

**Verdict: PASS ✅**

---

## Overall Summary

| Coder | Pass/Fail | Notes |
|-------|-----------|-------|
| **minimax** | ✅ PASS | 3 regimes, 35 transitions, bear/bull/sideways |
| **glm** | ✅ PASS | 4 regimes, 5 transitions, bear/crisis/bull/calm |
| **mimo** | ✅ PASS | 3 regimes, 30 transitions, bear/sideways/bull |
| **deepseek** | ✅ PASS | 2 regimes, 26 transitions, bull/bear. Only coder with auto-sys.path |
| **kimi** | ✅ PASS | 2 regimes (fixed), 2 transitions, sideways/bull. Was blocked, now resolved |
| **qwen** | ✅ PASS | 3 regimes, 46 transitions, bull/bear/sideways. Best regime balance |

### Acceptance Checklist

| # | Item | Status |
|---|------|--------|
| 1 | Code runs end-to-end without manual edits | ✅ All 6 run (5 need PYTHONPATH env var) |
| 2 | Output files created every time | ✅ CSV (1168 rows) + JSON for all 6 |
| 3 | Output is internally consistent | ✅ All have contiguous zero-based regime IDs, matching metadata |
| 4 | Evaluation window has meaningful regime variation | ✅ All detect regime changes in 2021-2025 |
| 5 | README explains exact deviations from the paper | ✅ All document signature simplification, threshold changes, and clustering methods |

### Overall Verdict: ALL 6 CODERS PASS ✅

### Recommendation

All 6 coders are approved. The previous blocking issue (coder-kimi's non-zero-based regime IDs and num_regimes mismatch) has been fixed and verified. No further revisions needed.

### Detailed Results Table

| Coder | Rows | Regimes | Transitions | Rate | Conf Range | Data End | Labels |
|-------|------|---------|-------------|------|------------|----------|--------|
| minimax | 1168 | 3 | 35 | 3.0% | 0.16–0.99 | 2025-12-31 | bear/sideways/bull |
| glm | 1168 | 4 | 5 | 0.4% | 0.50–0.998 | 2025-12-31 | bear/crisis/bull/calm |
| mimo | 1168 | 3 | 30 | 2.6% | 0.10–1.00 | 2025-12-31 | bear/sideways/bull |
| deepseek | 1168 | 2 | 26 | 2.2% | 0.30–0.90 | 2025-12-31 | bull/bear |
| kimi | 1168 | 2 | 2 | 0.2% | 0.65–0.86 | 2025-12-31 | sideways/bull |
| qwen | 1168 | 3 | 46 | 3.9% | 0.00–0.92 | 2025-12-31 | bull/bear/sideways |
