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
        self.model_manager = model_manager

    def act(self, state):
        close_history = pd.Series(state['close_history'])
        
        # Ensure we have enough data to calculate Bollinger Bands
        if len(close_history) < self.window:
            return 0  # Default action if not enough data

        upper_band, lower_band = self.calculate_bollinger_bands(close_history)

        # Ensure the bands have at least one value to check
        if len(upper_band) > 0 and len(close_history) > 0:
            action = 1 if close_history.iloc[-1] > upper_band.iloc[-1] else -1
        else:
            action = 0  # Default action if no valid data

        return action

    def calculate_bollinger_bands(self, close_prices):
        rolling_mean = close_prices.rolling(window=self.window).mean()
        rolling_std = close_prices.rolling(window=self.window).std()

        upper_band = rolling_mean + (rolling_std * self.num_std_dev)
        lower_band = rolling_mean - (rolling_std * self.num_std_dev)

        return upper_band, lower_band

    def save_model(self, filepath):
        model_data = {
            "window": self.window,
            "num_std_dev": self.num_std_dev
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.window = model_data["window"]
        self.num_std_dev = model_data["num_std_dev"]

    def train(self, experience):
        # Placeholder for training logic
        pass
