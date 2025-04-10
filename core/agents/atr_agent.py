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
        self.required_window = self.window
        self.model_manager = model_manager

    def act(self, state):
        high = pd.Series(state['high_history'])
        low = pd.Series(state['low_history'])
        close = pd.Series(state['close_history'])

        if not all([
            self.has_sufficient_data(high),
            self.has_sufficient_data(low),
            self.has_sufficient_data(close)
        ]):
            print(f"[ATRAgent] Not enough data. Required: {self.required_window}")
            return None

        atr = self.calculate_atr(high, low, close)

        if len(atr.dropna()) == 0:
            return None

        return 1 if atr.iloc[-1] > self.atr_threshold else -1

    def calculate_atr(self, high, low, close):
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1)
        true_range = tr.max(axis=1)
        return true_range.rolling(window=self.window).mean()
    
    def get_state_id(self, close_history):
        # ATR needs high/low/close, so fake it for now using close
        series = pd.Series(close_history)
        return f"atr_{int(series.pct_change().std() * 10000)}"


    def save_model(self, filepath):
        joblib.dump({
            "window": self.window,
            "atr_threshold": self.atr_threshold
        }, filepath)

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.window = model_data["window"]
        self.atr_threshold = model_data["atr_threshold"]
        self.required_window = self.window


    def train(self, experience):
        # Placeholder for training logic
        # Add your training logic here, e.g., using reinforcement learning or other methods
        pass
