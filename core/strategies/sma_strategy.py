import pandas as pd
from core.strategies.base_strategy import BaseStrategy

class SMAStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.short_window = config.get("short_window", 10)
        self.long_window = config.get("long_window", 30)

    def generate_signal(self, data: pd.Series) -> int:
        close_prices = data["close_history"]
        if len(close_prices) < self.long_window + 1:
            return 0

        short_sma = close_prices.rolling(window=self.short_window).mean()
        long_sma = close_prices.rolling(window=self.long_window).mean()

        if short_sma.iloc[-2] < long_sma.iloc[-2] and short_sma.iloc[-1] > long_sma.iloc[-1]:
            return 1  # Golden cross (buy)
        elif short_sma.iloc[-2] > long_sma.iloc[-2] and short_sma.iloc[-1] < long_sma.iloc[-1]:
            return -1  # Death cross (sell)
        else:
            return 0  # Hold
