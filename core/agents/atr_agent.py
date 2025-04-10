# core/agents/atr_agent.py
import pandas as pd
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager
import joblib

class ATRAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config)
        self.window = config['window']
        self.atr_threshold = config['atr_threshold']
        self.model_manager = model_manager

    def act(self, state):
        high_history = pd.Series(state['high_history'])
        low_history = pd.Series(state['low_history'])
        close_history = pd.Series(state['close_history'])

        # Ensure we have enough data to calculate ATR
        if len(high_history) < self.window or len(low_history) < self.window or len(close_history) < self.window:
            return 0  # Default action if not enough data

        atr = self.calculate_atr(high_history, low_history, close_history)

        # Ensure ATR has at least one value to check
        if len(atr) > 0:
            action = 1 if atr.iloc[-1] > self.atr_threshold else -1
        else:
            action = 0  # Default action if no valid ATR data

        return action

    def calculate_atr(self, high, low, close):
        tr = pd.concat([high - low, high - close.shift(), close.shift() - low], axis=1)
        true_range = tr.max(axis=1)
        atr = true_range.rolling(window=self.window).mean()
        return atr

    def save_model(self, filepath):
        model_data = {
            "window": self.window,
            "atr_threshold": self.atr_threshold
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.window = model_data["window"]
        self.atr_threshold = model_data["atr_threshold"]

    def train(self, experience):
        # Placeholder for training logic
        print("Training ATR agent...")
        # Add your training logic here, e.g., using reinforcement learning or other methods
        pass
