# core/agents/rsi_agent.py
import numpy as np
import pandas as pd
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager
import joblib

class RSIAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config)
        self.rsi_period = config['rsi_period']
        self.model_manager = model_manager

    def act(self, state):
        close_history = pd.Series(state['close_history'])
        
        # Ensure we have enough data to calculate RSI
        if len(close_history) < self.rsi_period:
            return 0  # Default action if not enough data

        rsi = self.calculate_rsi(close_history)
        
        # Check if the RSI series has data
        if len(rsi) > 0:
            action = 1 if rsi.iloc[-1] < 30 else -1  # Using iloc for better indexing
        else:
            action = 0  # Default action if no valid RSI data
        
        return action

    def calculate_rsi(self, close_prices):
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def save_model(self, filepath):
        model_data = {
            "rsi_period": self.rsi_period
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.rsi_period = model_data["rsi_period"]

    def train(self, experience):
        # Placeholder for training logic
        pass
