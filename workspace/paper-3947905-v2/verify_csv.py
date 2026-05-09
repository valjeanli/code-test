import pandas as pd
import numpy as np

df = pd.read_csv('/opt/data/kanban/workspaces/t_075f719e/paper-3947905-v2/signature_mmd_3regime_v2_regimes.csv')
df['date'] = pd.to_datetime(df['date'])

print("=== CSV VALIDATION ===")
print(f"Rows: {len(df)}")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Columns: {list(df.columns)}")
print()
print("=== predicted_regime ===")
print(df['predicted_regime'].value_counts().sort_index())
print(f"Unique: {sorted(df['predicted_regime'].unique())}")
print(f"Contiguous: {sorted(df['predicted_regime'].unique()) == list(range(df['predicted_regime'].max()+1))}")
print()
print("=== confidence ===")
print(f"min={df['confidence'].min():.4f} max={df['confidence'].max():.4f} mean={df['confidence'].mean():.4f} std={df['confidence'].std():.4f}")
print(f"In [0,1]: {(df['confidence'] >= 0).all() and (df['confidence'] <= 1).all()}")
print()
print("=== transition_flag ===")
print(f"Unique values: {df['transition_flag'].unique()}")
print(f"First value: {df['transition_flag'].iloc[0]} (should be 0)")
print(f"Total transitions: {df['transition_flag'].sum()}")
print()
print("=== regime_label ===")
print(df.groupby(['predicted_regime', 'regime_label']).size())
print()
print("=== Sample rows near transitions ===")
transitions = df[df['transition_flag'] == 1].head(10)
print(transitions[['date', 'predicted_regime', 'confidence', 'transition_flag', 'regime_label']])
print()
print("=== Dwell time analysis ===")
regimes = df['predicted_regime'].values
dwells = []
current, dwell = regimes[0], 1
for i in range(1, len(regimes)):
    if regimes[i] == current:
        dwell += 1
    else:
        dwells.append(dwell)
        current = regimes[i]
        dwell = 1
dwells.append(dwell)
print(f"Dwell times: count={len(dwells)}, min={min(dwells)}, max={max(dwells)}, avg={np.mean(dwells):.1f}")
print(f"Transitions: {len(dwells)-1}")
print(f"Transition rate: {len(dwells)-1}/{len(df)-1} = {(len(dwells)-1)/(len(df)-1):.3f}")
