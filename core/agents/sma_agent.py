# core/agents/sma_agent.py
import pandas as pd
import joblib
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager

class SMAAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config)
        self.short_window = config.get("short_window", 10)
        self.long_window = config.get("long_window", 30)
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 0.2)
        self.required_window = self.long_window + 1
        self.model_manager = model_manager

        self.q_table = {}  # state_id: [q_sell, q_hold, q_buy]

    def act(self, state):
        close_prices = pd.Series(state["close_history"])
        if not self.has_sufficient_data(close_prices):
            return None

        state_id = self.get_state_id(close_prices)
        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if pd.isna(close_prices).any():
            return None

        if pd.Series(close_prices).rolling(self.short_window).mean().isna().any():
            return None

        # Epsilon-greedy
        import numpy as np
        if np.random.rand() < self.epsilon:
            return np.random.choice([-1, 0, 1])

        action_index = int(np.argmax(self.q_table[state_id]))
        return [-1, 0, 1][action_index]

    def get_state_id(self, close_history):
        series = pd.Series(close_history)
        short_sma = series.rolling(window=self.short_window).mean()
        long_sma = series.rolling(window=self.long_window).mean()
        diff = short_sma.iloc[-1] - long_sma.iloc[-1] if not pd.isna(short_sma.iloc[-1]) else 0
        return f"sma_{int(diff * 100)}"

    def train(self, experience):
        state_id, action, reward, next_state_id = experience

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if next_state_id not in self.q_table:
            self.q_table[next_state_id] = [0, 0, 0]

        action_index = action + 1  # -1 => 0, 0 => 1, 1 => 2
        old_q = self.q_table[state_id][action_index]
        next_max_q = max(self.q_table[next_state_id])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state_id][action_index] = new_q

    def save_model(self, filepath):
        joblib.dump({"config": self.config, "q_table": self.q_table}, filepath)

    def load_model(self, filepath):
        data = joblib.load(filepath)
        self.config = data["config"]
        self.q_table = data.get("q_table", {})
        self.short_window = self.config.get("short_window", 10)
        self.long_window = self.config.get("long_window", 30)
        self.required_window = self.long_window + 1