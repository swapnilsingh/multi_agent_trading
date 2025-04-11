# core/agents/macd_agent.py
import pandas as pd
import joblib
import numpy as np
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager

class MACDAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config, model_manager)
        self.short_period = config['short_period']
        self.long_period = config['long_period']
        self.signal_period = config['signal_period']
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 0.2)
        self.required_window = self.long_period + self.signal_period
        self.q_table = {}
        self.model_manager = model_manager

    def act(self, state):
        close_prices = pd.Series(state.get("close_history", []))
        if len(close_prices) < self.required_window:
            return 0  # hold

        macd, signal, _ = self.calculate_macd(close_prices)
        macd_val = macd.iloc[-1] if not macd.empty else 0
        signal_val = signal.iloc[-1] if not signal.empty else 0

        state_dict = {"macd": macd_val, "signal_line": signal_val}
        state_id = self.get_state_id(state_dict)

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if np.random.rand() < self.epsilon:
            return np.random.choice([-1, 0, 1])

        action_index = int(np.argmax(self.q_table[state_id]))
        return [-1, 0, 1][action_index]

    def get_state_id(self, state):
        try:
            macd_val = float(state.get("macd", 0))
            signal_val = float(state.get("signal_line", 0))
            macd_delta = macd_val - signal_val
            return f"macd_{int(macd_delta * 100)}"
        except Exception as e:
            print(f"[{self.name}] Error computing state_id: {e}")
            return "macd_error"

    def learn(self, state_id, action, reward, next_state_id):
        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if next_state_id not in self.q_table:
            self.q_table[next_state_id] = [0, 0, 0]

        action_index = action + 1
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
