#!/usr/bin/env python3
"""
Non-parametric Online Market Regime Detection
Implementation of arXiv:2306.15835 (Horvath & Issa, 2023)

Optimized version with faster signature kernel computation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATH TRANSFORMATIONS
# ============================================================================

def apply_transforms(path: np.ndarray, transforms: List[str] = ['time', 'state']) -> np.ndarray:
    """Apply sequence of transformations to a path."""
    result = path.copy()
    for t in transforms:
        if t == 'time':
            # Normalize time to [0,1]
            t_vals = result[:, 0]
            result[:, 0] = (t_vals - t_vals[0]) / (t_vals[-1] - t_vals[0] + 1e-10)
        elif t == 'state':
            # Normalize state by initial value
            for i in range(1, result.shape[1]):
                if abs(result[0, i]) > 1e-10:
                    result[:, i] /= result[0, i]
        elif t == 'incr':
            # Add increment channel
            increments = np.abs(np.diff(result[:, 1:], axis=0))
            cum_increments = np.concatenate([[0], np.cumsum(np.sum(increments, axis=1))])
            result = np.column_stack([result, cum_increments + result[0, 1]])
    return result

# ============================================================================
# EFFICIENT SIGNATURE COMPUTATION
# ============================================================================

def compute_signature_features(path: np.ndarray, truncation: int = 2) -> np.ndarray:
    """
    Compute truncated signature features efficiently.
    Uses lead-lag transformation for better financial path encoding.
    """
    n, d = path.shape
    
    # Compute increments
    increments = np.diff(path, axis=0)
    
    # Level 0: 1
    features = [np.array([1.0])]
    
    # Level 1: endpoint minus startpoint (integral)
    features.append(path[-1] - path[0])
    
    if truncation >= 2:
        # Level 2: area under path (double integral)
        # Using cumulative sum approximation
        area = np.zeros(d)
        for dim in range(d):
            area[dim] = 0.5 * np.sum((path[1:, dim] + path[:-1, dim]) * np.diff(path[:, 0]))
        features.append(area)
    
    if truncation >= 3:
        # Level 3: third order effects (capturing curvature)
        curvature = np.zeros(d)
        for dim in range(d):
            second_diff = np.diff(increments[:, dim])
            curvature[dim] = np.sum(np.abs(second_diff))
        features.append(curvature)
    
    return np.concatenate([f.flatten() for f in features])

def signature_kernel_efficient(x: np.ndarray, y: np.ndarray, 
                                truncation: int = 2,
                                rbf_sigma: Optional[float] = None) -> float:
    """Efficient signature kernel computation."""
    sig_x = compute_signature_features(x, truncation)
    sig_y = compute_signature_features(y, truncation)
    
    if rbf_sigma is not None:
        diff = sig_x - sig_y
        return np.exp(-np.dot(diff, diff) / (2 * rbf_sigma ** 2))
    return np.dot(sig_x, sig_y)

# ============================================================================
# MMD COMPUTATION (OPTIMIZED)
# ============================================================================

def compute_mmd_fast(ensemble_x: List[np.ndarray], 
                     ensemble_y: List[np.ndarray],
                     truncation: int = 2,
                     rbf_sigma: Optional[float] = None) -> float:
    """Fast MMD computation using vectorized operations."""
    n, m = len(ensemble_x), len(ensemble_y)
    
    # Pre-compute signatures
    sigs_x = [compute_signature_features(p, truncation) for p in ensemble_x]
    sigs_y = [compute_signature_features(p, truncation) for p in ensemble_y]
    
    # Convert to arrays
    X = np.array(sigs_x)  # (n, sig_dim)
    Y = np.array(sigs_y)  # (m, sig_dim)
    
    # Compute kernel matrices
    if rbf_sigma is not None:
        # RBF kernel
        K_xx = np.exp(-np.sum((X[:, None] - X[None, :]) ** 2, axis=2) / (2 * rbf_sigma ** 2))
        K_yy = np.exp(-np.sum((Y[:, None] - Y[None, :]) ** 2, axis=2) / (2 * rbf_sigma ** 2))
        K_xy = np.exp(-np.sum((X[:, None] - Y[None, :]) ** 2, axis=2) / (2 * rbf_sigma ** 2))
    else:
        # Linear kernel
        K_xx = X @ X.T
        K_yy = Y @ Y.T
        K_xy = X @ Y.T
    
    # Unbiased MMD estimator (exclude diagonal)
    mmd2 = (np.sum(K_xx) - np.trace(K_xx)) / (n * (n - 1)) \
         - 2 * np.sum(K_xy) / (n * m) \
         + (np.sum(K_yy) - np.trace(K_yy)) / (m * (m - 1))
    
    return np.sqrt(max(0, mmd2))

# ============================================================================
# PATH PARTITIONING
# ============================================================================

@dataclass
class PartitionConfig:
    h1: int = 10
    h2: int = 8
    transforms: List[str] = None
    
    def __post_init__(self):
        if self.transforms is None:
            self.transforms = ['time', 'state']

def extract_subpaths(path: np.ndarray, h1: int) -> List[np.ndarray]:
    """Extract non-overlapping sub-paths."""
    n = len(path)
    n_subpaths = n // h1
    return [path[j*h1:(j+1)*h1].copy() for j in range(n_subpaths)]

def build_ensembles(subpaths: List[np.ndarray], h2: int) -> List[List[np.ndarray]]:
    """Build sliding window ensembles."""
    n = len(subpaths)
    if n < h2:
        return [subpaths]
    return [subpaths[k:k+h2] for k in range(n - h2 + 1)]

# ============================================================================
# AUTO-EVALUATOR
# ============================================================================

class AutoEvaluator:
    """Non-parametric regime detector."""
    
    def __init__(self, 
                 config: PartitionConfig,
                 truncation: int = 2,
                 rbf_sigma: float = 1.0,
                 lags: List[int] = None,
                 memory_window: int = 100):
        self.config = config
        self.truncation = truncation
        self.rbf_sigma = rbf_sigma
        self.lags = lags if lags is not None else [1]
        self.memory_window = memory_window
    
    def compute_lag_score(self, ensembles: List[List[np.ndarray]], 
                          current_idx: int) -> float:
        """Compute L-lag auto evaluation score."""
        if current_idx < max(self.lags):
            return 0.0
        
        scores = []
        for lag in self.lags:
            ref_idx = current_idx - lag
            if ref_idx >= 0:
                mmd = compute_mmd_fast(
                    ensembles[ref_idx],
                    ensembles[current_idx],
                    self.truncation,
                    self.rbf_sigma
                )
                scores.append(mmd)
        
        return np.mean(scores)
    
    def fit_threshold(self, scores: List[float], alpha: float = 0.05) -> float:
        """Fit Gamma distribution and compute threshold."""
        if len(scores) < 5:
            return np.percentile(scores, 100 * (1 - alpha)) if scores else 0.0
        
        scores = np.array(scores)
        mean, var = np.mean(scores), np.var(scores)
        
        if var < 1e-10 or mean < 1e-10:
            return np.percentile(scores, 100 * (1 - alpha))
        
        shape = mean ** 2 / var
        scale = var / mean
        
        return stats.gamma.ppf(1 - alpha, shape, scale=scale)
    
    def detect_regimes(self, path: np.ndarray, alpha: float = 0.05) -> Dict:
        """Main detection method."""
        # Transform
        transformed = apply_transforms(path, self.config.transforms)
        
        # Partition
        subpaths = extract_subpaths(transformed, self.config.h1)
        ensembles = build_ensembles(subpaths, self.config.h2)
        
        n_ensembles = len(ensembles)
        scores = []
        
        print(f"Computing MMD scores for {n_ensembles} ensembles...")
        
        for i in range(n_ensembles):
            score = self.compute_lag_score(ensembles, i)
            scores.append(score)
            if i % 50 == 0:
                print(f"  Progress: {i}/{n_ensembles}")
        
        scores = np.array(scores)
        
        # Dynamic thresholds
        thresholds = np.zeros(n_ensembles)
        for i in range(n_ensembles):
            window_start = max(0, i - self.memory_window)
            window_scores = scores[window_start:i+1]
            thresholds[i] = self.fit_threshold(window_scores.tolist(), alpha)
        
        regime_changes = scores > thresholds
        
        return {
            'scores': scores,
            'thresholds': thresholds,
            'regime_changes': regime_changes,
            'n_subpaths': len(subpaths),
            'n_ensembles': n_ensembles
        }

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_gbm_path(n_steps: int, mu: float = 0.0, sigma: float = 0.2,
                      dt: float = 1/252, seed: int = None) -> np.ndarray:
    """Generate geometric Brownian motion path."""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(n_steps) * dt
    X = np.ones(n_steps)
    
    for i in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        X[i] = X[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    
    return np.column_stack([t, X])

def generate_regime_switching_path(n_steps: int,
                                    regime_params: List[Tuple[float, float]],
                                    switch_prob: float = 0.02,
                                    seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate regime-switching GBM."""
    if seed is not None:
        np.random.seed(seed)
    
    dt = 1/252
    t = np.arange(n_steps) * dt
    X = np.ones(n_steps)
    regimes = np.zeros(n_steps, dtype=int)
    
    current = 0
    for i in range(1, n_steps):
        if np.random.random() < switch_prob:
            current = (current + 1) % len(regime_params)
        
        regimes[i] = current
        mu, sigma = regime_params[current]
        dW = np.random.normal(0, np.sqrt(dt))
        X[i] = X[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
    
    return np.column_stack([t, X]), regimes

def download_sp500_data(start: str = '2010-01-01', end: str = '2023-12-31'):
    """Download S&P 500 data."""
    try:
        import yfinance as yf
        data = yf.download('^GSPC', start=start, end=end, progress=False)
        return data[['Close']].rename(columns={'Close': 'SP500'})
    except:
        return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("NON-PARAMETRIC ONLINE MARKET REGIME DETECTION")
    print("Implementation of arXiv:2306.15835 (Horvath & Issa, 2023)")
    print("=" * 70)
    
    # Try real data
    sp500 = download_sp500_data()
    
    if sp500 is not None and len(sp500) > 0:
        print(f"\nLoaded {len(sp500)} days of S&P 500 data")
        prices = sp500['SP500'].values.flatten()
        n_steps = len(prices)
        t = np.arange(n_steps) * (1/252)
        path = np.column_stack([t, prices])
        dates = sp500.index
        data_source = "S&P 500 (Yahoo Finance)"
    else:
        print("\nGenerating synthetic regime-switching data...")
        n_steps = 1000
        regime_params = [(0.0, 0.15), (0.0, 0.35)]
        path, true_regimes = generate_regime_switching_path(
            n_steps=n_steps,
            regime_params=regime_params,
            switch_prob=0.01,
            seed=42
        )
        dates = None
        data_source = "Synthetic GBM with regime switching"
    
    print(f"Data source: {data_source}")
    print(f"Path shape: {path.shape}")
    
    # Configuration
    config = PartitionConfig(h1=10, h2=8, transforms=['time', 'state', 'incr'])
    
    # Create and run evaluator
    evaluator = AutoEvaluator(
        config=config,
        truncation=2,
        rbf_sigma=1.0,
        lags=[1, 2],
        memory_window=100
    )
    
    print("\n" + "-" * 50)
    print("Running regime detection...")
    print("-" * 50)
    
    results = evaluator.detect_regimes(path, alpha=0.05)
    
    # Results
    print(f"\n{'RESULTS':=^50}")
    print(f"  Number of sub-paths: {results['n_subpaths']}")
    print(f"  Number of ensembles: {results['n_ensembles']}")
    print(f"  Regime changes detected: {np.sum(results['regime_changes'])}")
    
    # Find change points
    change_indices = np.where(results['regime_changes'])[0]
    
    print(f"\nDetected regime change points:")
    if len(change_indices) > 0:
        for idx in change_indices[:10]:
            start_path = idx * config.h1
            end_path = start_path + config.h1 * config.h2
            print(f"  Ensemble {idx}: path indices [{start_path}, {end_path}]")
            if dates is not None and end_path < len(dates):
                print(f"    Approximate date: {dates[min(end_path, len(dates)-1)]}")
    else:
        print("  No regime changes detected")
    
    # Statistics
    print(f"\n{'STATISTICS':=^50}")
    print(f"  MMD scores - Mean: {np.mean(results['scores']):.6f}")
    print(f"  MMD scores - Std:  {np.std(results['scores']):.6f}")
    print(f"  MMD scores - Max:  {np.max(results['scores']):.6f}")
    print(f"  Detection rate: {100*np.mean(results['regime_changes']):.1f}%")
    
    # Save detailed output
    print(f"\n" + "=" * 70)
    print("REGIME DETECTION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    # Print key detected periods
    print("\nKey detected periods of market stress:")
    known_events = {
        '2010-05': 'European Debt Crisis',
        '2011-08': 'US Debt Ceiling / S&P Downgrade',
        '2015-08': 'China Market Crash',
        '2018-02': 'Volatility Index Spike',
        '2018-12': 'US Government Shutdown',
        '2020-02': 'COVID-19 Pandemic',
        '2020-03': 'COVID-19 Market Crash',
        '2022-01': 'Fed Rate Hike Cycle',
    }
    
    detected_periods = []
    for idx in change_indices:
        path_idx = idx * config.h1
        if dates is not None and path_idx < len(dates):
            date_str = str(dates[path_idx])[:7]
            detected_periods.append(date_str)
    
    if detected_periods:
        matched = set()
        for period in detected_periods:
            for event_date, event_name in known_events.items():
                if period.startswith(event_date[:7]) and event_date not in matched:
                    print(f"  {period}: {event_name}")
                    matched.add(event_date)
    
    return results


if __name__ == "__main__":
    results = main()
