import pandas as pd
from core.strategies.base_strategy import BaseStrategy

class MACDStrategy(BaseStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.fast_period = config.get("fast_period", 12)
        self.slow_period = config.get("slow_period", 26)
        self.signal_period = config.get("signal_period", 9)

    def generate_signal(self, data: pd.Series) -> int:
        close_prices = data["close_history"]
        if len(close_prices) < self.slow_period + self.signal_period:
            return 0

        ema_fast = close_prices.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close_prices.ewm(span=self.slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()

        if macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
            return 1  # Bullish crossover
        elif macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
            return -1  # Bearish crossover
        else:
            return 0
