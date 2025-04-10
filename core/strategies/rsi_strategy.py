import pandas as pd
from core.strategies.base_strategy import BaseStrategy

class RSIStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.period = config.get("period", 14)
        self.lower_threshold = config.get("lower_threshold", 30)
        self.upper_threshold = config.get("upper_threshold", 70)

    def generate_signal(self, data: pd.Series) -> int:
        close_prices = data["close_history"]
        if len(close_prices) < self.period + 1:
            return 0

        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        if current_rsi < self.lower_threshold:
            return 1  # Buy
        elif current_rsi > self.upper_threshold:
            return -1  # Sell
        else:
            return 0  # Hold
