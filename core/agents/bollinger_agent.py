# core/agents/bollinger_agent.py
import pandas as pd
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager
import joblib

class BollingerAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config)
        self.window = config['window']
        self.num_std_dev = config['num_std_dev']
        self.required_window = self.window
        self.model_manager = model_manager

    def act(self, state):
        close_history = pd.Series(state['close_history'])

        if not self.has_sufficient_data(close_history):
            print(f"[BollingerAgent] Not enough data. Required: {self.required_window}, Got: {len(close_history)}")
            return None

        upper_band, lower_band = self.calculate_bollinger_bands(close_history)

        if pd.notna(upper_band.iloc[-1]) and pd.notna(close_history.iloc[-1]):
            return 1 if close_history.iloc[-1] > upper_band.iloc[-1] else -1
        else:
            return None

    def calculate_bollinger_bands(self, close_prices):
        mean = close_prices.rolling(window=self.window).mean()
        std = close_prices.rolling(window=self.window).std()
        upper = mean + (self.num_std_dev * std)
        lower = mean - (self.num_std_dev * std)
        return upper, lower
    
    def get_state_id(self, close_history):
        close_series = pd.Series(close_history)
        upper, lower = self.calculate_bollinger_bands(close_series)
        price = close_series.iloc[-1]
        if pd.isna(upper.iloc[-1]) or pd.isna(lower.iloc[-1]):
            return "neutral"
        position = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-5)
        return f"bb_{int(position * 100)}"



    def save_model(self, filepath):
        joblib.dump({
            "window": self.window,
            "num_std_dev": self.num_std_dev
        }, filepath)

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.window = model_data["window"]
        self.num_std_dev = model_data["num_std_dev"]
        self.required_window = self.window


    def train(self, experience):
        # Placeholder for training logic
        pass
