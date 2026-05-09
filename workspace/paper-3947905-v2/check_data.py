import pandas as pd
df = pd.read_parquet('/opt/data/kanban/workspaces/t_f9e0a695/sp500_data/sp500_ohlcv_20yr.parquet')
df = df.reset_index()
print('Date range:', df['date'].min(), '->', df['date'].max())
print('Total rows:', len(df))
future = df[df['date'] >= '2026-01-01']
print('Future dates (2026+):', len(future))
