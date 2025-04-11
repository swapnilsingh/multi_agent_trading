# core/agents/sma_agent.py
import pandas as pd
import numpy as np
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
        self.model_manager = model_manager

        self.required_window = self.long_window + 1
        self.q_table = {}  # state_id -> [q_sell, q_hold, q_buy]

    def act(self, state):
        close_prices = pd.Series(state["close_history"])

        if not self.has_sufficient_data(close_prices):
            return None

        short_sma = close_prices.rolling(window=self.short_window).mean()
        long_sma = close_prices.rolling(window=self.long_window).mean()

        state_id = self.get_state_id(close_prices)
        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if np.random.rand() < self.epsilon:
            return np.random.choice([-1, 0, 1])

        action_index = int(np.argmax(self.q_table[state_id]))
        return [-1, 0, 1][action_index]

    def get_state_id(self, close_prices):
        close_prices = pd.Series(close_prices)  # Ensure it's a pandas Series
        short_sma = close_prices.rolling(window=self.short_window).mean()
        long_sma = close_prices.rolling(window=self.long_window).mean()

        prev_short = short_sma.iloc[-2]
        prev_long = long_sma.iloc[-2]
        curr_short = short_sma.iloc[-1]
        curr_long = long_sma.iloc[-1]

        if pd.isna(prev_short) or pd.isna(prev_long) or pd.isna(curr_short) or pd.isna(curr_long):
            return "sma_nan"

        crossover = 0
        if prev_short < prev_long and curr_short > curr_long:
            crossover = 1  # Bullish crossover
        elif prev_short > prev_long and curr_short < curr_long:
            crossover = -1  # Bearish crossover

        return f"sma_{crossover}"

    def train(self, experience):
        state_id, action, reward, next_state_id = experience

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if next_state_id not in self.q_table:
            self.q_table[next_state_id] = [0, 0, 0]

        action_index = action + 1
        old_q = self.q_table[state_id][action_index]
        next_max_q = max(self.q_table[next_state_id])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state_id][action_index] = new_q

    def save_model(self, filepath):
        joblib.dump({
            "short_window": self.short_window,
            "long_window": self.long_window,
            "q_table": self.q_table
        }, filepath)

    def load_model(self, filepath):
        data = joblib.load(filepath)
        self.short_window = data["short_window"]
        self.long_window = data["long_window"]
        self.q_table = data.get("q_table", {})
        self.required_window = self.long_window + 1
