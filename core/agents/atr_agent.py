import pandas as pd
from core.agents.base_agent import BaseAgent

class ATRAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.window = config.get("window", 14)
        self.multiplier = config.get("multiplier", 1.5)

    def act(self, state: pd.Series) -> int:
        # state must include history of highs, lows, closes
        high = state["high_history"]
        low = state["low_history"]
        close = state["close_history"]

        if len(close) < self.window + 1:
            return 0  # Not enough data

        prev_close = close.shift(1)

        tr = pd.concat([
            (high - low),
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

    def train(self, experience):
        pass  # No training for rule-based agent

    def save(self, filepath):
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(self.config, f)

    def load(self, filepath):
        import pickle
        with open(filepath, "rb") as f:
            self.config = pickle.load(f)
