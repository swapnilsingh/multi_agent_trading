import pandas as pd
from core.agents.base_agent import BaseAgent

class SMAAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.short_window = config.get("short_window", 10)
        self.long_window = config.get("long_window", 30)

    def act(self, state: pd.Series) -> int:
        close_prices = state["close_history"]
        if len(close_prices) < self.long_window + 1:
            return 0  # Not enough data

        short_sma = close_prices.rolling(window=self.short_window).mean()
        long_sma = close_prices.rolling(window=self.long_window).mean()

        # Detect crossover
        if short_sma.iloc[-2] < long_sma.iloc[-2] and short_sma.iloc[-1] > long_sma.iloc[-1]:
            return 1  # Buy
        elif short_sma.iloc[-2] > long_sma.iloc[-2] and short_sma.iloc[-1] < long_sma.iloc[-1]:
            return -1  # Sell
        else:
            return 0  # Hold

    def train(self, experience):
        pass

    def save(self, filepath):
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(self.config, f)

    def load(self, filepath):
        import pickle
        with open(filepath, "rb") as f:
            self.config = pickle.load(f)
