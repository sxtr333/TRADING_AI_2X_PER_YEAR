import pandas as pd
df = pd.read_parquet("model6/data/BTCUSDT_15m_features.parquet")
print("Строк/столбцов:", df.shape)
print("Колонки:")
for i, c in enumerate(df.columns):
    print(f"{i:02d}: {c}")

import pandas as pd
df = pd.read_parquet("model6/data/BTCUSDT_15m_features.parquet")
  # первые 10 строк по всем колонкам
print(df.head(10))
  # если нужно отдельно по каждой колонке:
for col in df.columns:
    print(f"\n=== {col} ===")
    print(df[col].head(10).to_string(index=False))