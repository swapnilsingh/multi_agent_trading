import pandas as pd
from core.strategies.base_strategy import BaseStrategy

class BollingerStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.window = config.get("window", 20)
        self.std_dev = config.get("std_dev", 2)

    def generate_signal(self, data: pd.Series) -> int:
        close_prices = data["close_history"]
        if len(close_prices) < self.window:
            return 0

        sma = close_prices.rolling(window=self.window).mean()
        std = close_prices.rolling(window=self.window).std()
        upper_band = sma + self.std_dev * std
        lower_band = sma - self.std_dev * std

        current_price = close_prices.iloc[-1]
        upper = upper_band.iloc[-1]
        lower = lower_band.iloc[-1]

        if current_price < lower:
            return 1  # Buy
        elif current_price > upper:
            return -1  # Sell
        else:
            return 0  # Hold
