import pandas as pd
import numpy as np

df = pd.read_csv('sig_mmd_regimes.csv')
print('Total rows:', len(df))
print('Regime distribution:')
print(df['predicted_regime'].value_counts().sort_index().to_string())
print()
print('Transition count:', int(df['transition_flag'].sum()))
rate = df['transition_flag'].sum() / (len(df)-1)
print(f'Transition rate: {rate:.4f}')
print()
labels = df['predicted_regime'].values
dwells = []
cur = labels[0]
cnt = 1
for i in range(1, len(labels)):
    if labels[i] == cur:
        cnt += 1
    else:
        dwells.append(cnt)
        cur = labels[i]
        cnt = 1
dwells.append(cnt)
print(f'Segments: {len(dwells)}')
print(f'Avg dwell: {np.mean(dwells):.1f} days')
print(f'Median dwell: {np.median(dwells):.0f} days')
print(f'Min/Max dwell: {min(dwells)}/{max(dwells)}')
print()
trans = df[df['transition_flag']==1].head(15)
for _, r in trans.iterrows():
    print(f"  {r['date']}: -> regime {r['predicted_regime']} ({r['regime_label']})")
