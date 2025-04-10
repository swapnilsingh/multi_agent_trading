# core/agents/macd_agent.py
import pandas as pd
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager
import joblib

class MACDAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config)
        self.short_period = config['short_period']
        self.long_period = config['long_period']
        self.signal_period = config['signal_period']
        self.model_manager = model_manager

    def act(self, state):
        close_history = pd.Series(state['close_history'])

        # Ensure we have enough data to calculate MACD
        if len(close_history) < self.long_period:
            return 0  # Default action if not enough data

        # Calculate MACD and signal
        macd, signal, _ = self.calculate_macd(close_history)

        # Ensure we have enough MACD data to access the last element
        if len(macd) > 0 and len(signal) > 0:
            action = 1 if macd.iloc[-1] > signal.iloc[-1] else -1
        else:
            action = 0  # Default action if no valid MACD data
        
        return action

    def calculate_macd(self, close_prices):
        short_ema = close_prices.ewm(span=self.short_period, adjust=False).mean()
        long_ema = close_prices.ewm(span=self.long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def save_model(self, filepath):
        model_data = {
            "short_period": self.short_period,
            "long_period": self.long_period,
            "signal_period": self.signal_period
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.short_period = model_data["short_period"]
        self.long_period = model_data["long_period"]
        self.signal_period = model_data["signal_period"]

    def train(self, experience):
        # Placeholder for training logic
        pass
