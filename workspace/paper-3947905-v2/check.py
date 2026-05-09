import pandas as pd
df = pd.read_csv('sig_mmd_regimes.csv')
for r in sorted(df['predicted_regime'].unique()):
    sub = df[df['predicted_regime']==r]
    print(f"Regime {r} ({sub['regime_label'].iloc[0]}): n={len(sub)}, conf=[{sub['confidence'].min():.3f}, {sub['confidence'].max():.3f}], mean={sub['confidence'].mean():.3f}")
print()
n_neg = (df['predicted_regime'] < 0).sum()
print(f"Negative regime values: {n_neg}")
print(f"First row transition_flag: {df['transition_flag'].iloc[0]}")
print(f"Confidence range: [{df['confidence'].min():.4f}, {df['confidence'].max():.4f}]")
