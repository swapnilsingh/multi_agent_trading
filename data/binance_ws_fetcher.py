# core/data/binance_rest_fetcher.py
import requests
import pandas as pd
from datetime import datetime

class BinanceRestFetcher:
    def __init__(self, symbol="BTCUSDT", interval="1m", limit=500):
        self.symbol = symbol.upper()
        self.interval = interval
        self.limit = limit
        self.endpoint = "https://api.binance.com/api/v3/klines"

    def fetch(self):
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": self.limit
        }

        print(f"[Binance REST] Fetching {self.limit} candles for {self.symbol}...")
        response = requests.get(self.endpoint, params=params)
        response.raise_for_status()

        raw_data = response.json()
        parsed_data = [
            {
                "timestamp": datetime.fromtimestamp(entry[0] / 1000),
                "Open": float(entry[1]),
                "High": float(entry[2]),
                "Low": float(entry[3]),
                "Close": float(entry[4]),
                "Volume": float(entry[5])
            }
            for entry in raw_data
        ]

        df = pd.DataFrame(parsed_data)
        return df

if __name__ == "__main__":
    fetcher = BinanceRestFetcher(symbol="BTCUSDT", interval="1m", limit=500)
    df = fetcher.fetch()
    df.to_csv("data/live_binance_ohlcv.csv", index=False)
    print("[Binance REST] Data saved to data/live_binance_ohlcv.csv")
