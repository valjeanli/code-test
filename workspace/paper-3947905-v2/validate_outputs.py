#!/usr/bin/env python3
"""Final validation of output files against spec requirements."""
import pandas as pd
import json
import sys
from pathlib import Path

output_dir = Path("/opt/data/kanban/workspaces/t_c36a4eed/paper-3947905-v2/outputs")
algo_name = "sigmmd_regime"

errors = []

# 1. Files exist
csv_path = output_dir / f"{algo_name}_regimes.csv"
json_path = output_dir / f"{algo_name}_metadata.json"
if not csv_path.exists():
    errors.append(f"CSV missing: {csv_path}")
if not json_path.exists():
    errors.append(f"JSON missing: {json_path}")

# 2. CSV validation
df = pd.read_csv(csv_path)
print(f"CSV rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Required columns
required = {"date", "predicted_regime", "confidence", "transition_flag"}
missing = required - set(df.columns)
if missing:
    errors.append(f"Missing columns: {missing}")

# Date format
try:
    dates = pd.to_datetime(df["date"])
    if not dates.is_monotonic_increasing:
        errors.append("Dates not sorted ascending")
    print(f"Date range: {dates.min().date()} to {dates.max().date()}")
except:
    errors.append("Date parsing failed")

# predicted_regime
regimes = df["predicted_regime"].values
unique_r = sorted(set(regimes))
print(f"Unique regimes: {unique_r}")
if not set(unique_r) == set(range(len(unique_r))):
    errors.append(f"Regimes not contiguous from 0: {unique_r}")
if any(r < 0 for r in regimes):
    errors.append("Negative regime IDs found")
if pd.isna(regimes).any():
    errors.append("NaN in predicted_regime")

# confidence
conf = df["confidence"].values
cmin, cmax = conf.min(), conf.max()
print(f"Confidence range: [{cmin:.4f}, {cmax:.4f}]")
if cmin < 0 or cmax > 1:
    errors.append(f"Confidence out of [0,1]: [{cmin},{cmax}]")

# transition_flag
tf = df["transition_flag"].values
unique_tf = set(tf)
print(f"Unique transition_flags: {unique_tf}")
if not unique_tf.issubset({0, 1}):
    errors.append(f"Non-binary transition_flag: {unique_tf}")
if len(df) > 0 and tf[0] != 0:
    errors.append(f"First row transition_flag={tf[0]}, expected 0")
n_trans = int(tf.sum())
print(f"Total transitions: {n_trans} ({n_trans/len(df)*100:.1f}%)")

# 3. JSON validation
with open(json_path) as f:
    meta = json.load(f)
print(f"Metadata keys: {list(meta.keys())}")

required_meta = {"algorithm", "algorithm_full_name", "algorithm_family", "paper_reference",
                 "num_regimes", "regime_labels_map", "parameters", "training_window_days",
                 "features_used", "generated_at", "output_spec_version"}
missing_meta = required_meta - set(meta.keys())
if missing_meta:
    errors.append(f"Missing metadata keys: {missing_meta}")

if meta["num_regimes"] != len(unique_r):
    errors.append(f"meta num_regimes={meta['num_regimes']} != actual {len(unique_r)}")

label_keys = set(meta["regime_labels_map"].keys())
expected_keys = set(str(r) for r in unique_r)
if label_keys != expected_keys:
    errors.append(f"label map keys mismatch: {label_keys} vs {expected_keys}")

# 4. File encoding check
with open(csv_path, "rb") as f:
    first_bytes = f.read(3)
    if first_bytes == b'\xef\xbb\xbf':
        errors.append("CSV has BOM")

print(f"\n{'='*50}")
if errors:
    print(f"❌ {len(errors)} VALIDATION ERRORS:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("✅ ALL VALIDATION CHECKS PASSED")
    print(f"   {len(df)} rows, {len(unique_r)} regimes, {n_trans} transitions")
    print(f"   Files: {csv_path.name}, {json_path.name}")
