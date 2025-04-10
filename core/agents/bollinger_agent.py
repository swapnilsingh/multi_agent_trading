import pandas as pd
from core.agents.base_agent import BaseAgent

class BollingerAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.window = config.get("window", 20)
        self.std_dev = config.get("std_dev", 2)

    def act(self, state: pd.Series) -> int:
        close_prices = state["close_history"]  # expects a pd.Series of recent closing prices
        if len(close_prices) < self.window:
            return 0  # Not enough data

        sma = close_prices.rolling(window=self.window).mean()
        std = close_prices.rolling(window=self.window).std()
        upper_band = sma + (self.std_dev * std)
        lower_band = sma - (self.std_dev * std)

        current_price = close_prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]

        if current_price < current_lower:
            return 1  # Buy
        elif current_price > current_upper:
            return -1  # Sell
        else:
            return 0  # Hold

    def train(self, experience):
        pass  # No training needed for rule-based agent

    def save(self, filepath):
        import pickle
        with open(filepath, "wb") as f:
            pickle.dump(self.config, f)

    def load(self, filepath):
        import pickle
        with open(filepath, "rb") as f:
            self.config = pickle.load(f)
