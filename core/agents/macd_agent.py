# core/agents/macd_agent.py
import pandas as pd
import joblib
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager

class MACDAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config)
        self.short_period = config['short_period']
        self.long_period = config['long_period']
        self.signal_period = config['signal_period']
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 0.2)
        self.model_manager = model_manager

        self.required_window = self.long_period + self.signal_period
        self.q_table = {}  # state_id -> [q_sell, q_hold, q_buy]

    def act(self, state):
        close_history = pd.Series(state['close_history'])

        if not self.has_sufficient_data(close_history):
            return None

        macd, signal, hist = self.calculate_macd(close_history)
        macd_delta = macd.iloc[-1] - signal.iloc[-1] if len(macd) > 0 and len(signal) > 0 else 0
        state_id = self.get_state_id(macd_delta)

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]  # sell, hold, buy

        import numpy as np
        if np.random.rand() < self.epsilon:
            return np.random.choice([-1, 0, 1])

        action_index = int(np.argmax(self.q_table[state_id]))
        return [-1, 0, 1][action_index]

    def get_state_id(self, macd_delta):
        if isinstance(macd_delta, list):
            macd_delta = macd_delta[-1]  # in case it's a list
        return f"macd_{int(float(macd_delta) * 100)}"

    def train(self, experience):
        state_id, action, reward, next_state_id = experience

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if next_state_id not in self.q_table:
            self.q_table[next_state_id] = [0, 0, 0]

        action_index = action + 1  # -1=>0, 0=>1, 1=>2
        old_q = self.q_table[state_id][action_index]
        next_max_q = max(self.q_table[next_state_id])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state_id][action_index] = new_q

    def calculate_macd(self, close_prices):
        short_ema = close_prices.ewm(span=self.short_period, adjust=False).mean()
        long_ema = close_prices.ewm(span=self.long_period, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def save_model(self, filepath):
        joblib.dump({
            "short_period": self.short_period,
            "long_period": self.long_period,
            "signal_period": self.signal_period,
            "q_table": self.q_table
        }, filepath)

    def load_model(self, filepath):
        data = joblib.load(filepath)
        self.short_period = data["short_period"]
        self.long_period = data["long_period"]
        self.signal_period = data["signal_period"]
        self.q_table = data.get("q_table", {})
        self.required_window = self.long_period + self.signal_period