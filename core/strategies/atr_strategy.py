import pandas as pd
from core.strategies.base_strategy import BaseStrategy

class ATRStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.window = config.get("window", 14)
        self.multiplier = config.get("multiplier", 1.5)

    def generate_signal(self, data: pd.Series) -> int:
        high = data["high_history"]
        low = data["low_history"]
        close = data["close_history"]

        if len(close) < self.window + 1:
            return 0

        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=self.window).mean()
        current_atr = atr.iloc[-1]
        price_change = close.iloc[-1] - close.iloc[-2]

        if price_change > current_atr * self.multiplier:
            return 1  # Buy
        elif price_change < -current_atr * self.multiplier:
            return -1  # Sell
        else:
            return 0  # Hold
