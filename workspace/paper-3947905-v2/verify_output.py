import pandas as pd
import json

CSV = "/opt/data/kanban/workspaces/t_1b79a957/paper-3947905-v2/sig_mmd_horvath_issa_regimes.csv"
JSON = "/opt/data/kanban/workspaces/t_1b79a957/paper-3947905-v2/sig_mmd_horvath_issa_metadata.json"

df = pd.read_csv(CSV)
with open(JSON) as f:
    meta = json.load(f)

print("=== CSV ===")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())
print(f"\nRegime distribution:")
print(df['predicted_regime'].value_counts().sort_index())
print(f"\nRegime labels:")
for _, row in df[['predicted_regime', 'regime_label']].drop_duplicates().sort_values('predicted_regime').iterrows():
    print(f"  Regime {row['predicted_regime']}: {row['regime_label']}")
print(f"\nTransitions: {df['transition_flag'].sum()} out of {len(df)}")
print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
print(f"\nConfidence stats:")
print(df['confidence'].describe())
print(f"\n=== METADATA ===")
print(json.dumps(meta, indent=2))

# Show transition dates
transitions = df[df['transition_flag'] == 1]
print(f"\n=== TRANSITION DATES ===")
for _, row in transitions.iterrows():
    print(f"  {row['date']}: regime {row['predicted_regime']} ({row['regime_label']})")