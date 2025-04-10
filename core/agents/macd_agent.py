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
        self.required_window = self.long_period  # Longest required
        self.model_manager = model_manager

    def act(self, state):
        close_history = pd.Series(state['close_history'])

        if not self.has_sufficient_data(close_history):
            print(f"[MACDAgent] Not enough data. Required: {self.required_window}, Got: {len(close_history)}")
            return None

        macd, signal, _ = self.calculate_macd(close_history)

        if len(macd.dropna()) > 0 and len(signal.dropna()) > 0:
            return 1 if macd.iloc[-1] > signal.iloc[-1] else -1
        else:
            return None

    def calculate_macd(self, close_prices):
        short_ema = close_prices.ewm(span=self.short_period, adjust=False).mean()
        long_ema = close_prices.ewm(span=self.long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
    
    def get_state_id(self, close_history):
        close_series = pd.Series(close_history)
        macd, signal, _ = self.calculate_macd(close_series)
        if len(macd.dropna()) == 0 or len(signal.dropna()) == 0:
            return "neutral"
        diff = macd.iloc[-1] - signal.iloc[-1]
        return f"macd_{int(diff * 100)}"



    def save_model(self, filepath):
        joblib.dump({
            "short_period": self.short_period,
            "long_period": self.long_period,
            "signal_period": self.signal_period
        }, filepath)

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.short_period = model_data["short_period"]
        self.long_period = model_data["long_period"]
        self.signal_period = model_data["signal_period"]
        self.required_window = self.long_period


    def train(self, experience):
        # Placeholder for training logic
        pass
