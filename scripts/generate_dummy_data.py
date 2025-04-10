# save this in scripts/generate_dummy_data.py
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
    "open": np.random.rand(100) * 100,
    "high": np.random.rand(100) * 100 + 5,
    "low": np.random.rand(100) * 100 - 5,
    "close": np.random.rand(100) * 100,
    "volume": np.random.randint(100, 1000, size=100)
})

df.to_csv("data/processed/btc_ohlcv.csv", index=False)
print("✅ Dummy data created.")
