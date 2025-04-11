# core/agents/bollinger_agent.py
import pandas as pd
import joblib
import numpy as np
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager

class BollingerAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config)
        self.window = config['window']
        self.num_std_dev = config['num_std_dev']
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 0.2)
        self.model_manager = model_manager

        self.required_window = self.window
        self.q_table = {}  # state_id -> [q_sell, q_hold, q_buy]

    def act(self, state):
        close_history = pd.Series(state['close_history'])

        if not self.has_sufficient_data(close_history):
            return None

        upper_band, lower_band = self.calculate_bollinger_bands(close_history)
        price = close_history.iloc[-1]

        # State is the distance of current price from the bands
        if len(upper_band) == 0 or len(lower_band) == 0:
            return None

        band_range = upper_band.iloc[-1] - lower_band.iloc[-1]
        if band_range == 0:
            return None

        relative_position = (price - lower_band.iloc[-1]) / band_range
        state_id = self.get_state_id(relative_position)

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]  # sell, hold, buy

        if np.random.rand() < self.epsilon:
            return np.random.choice([-1, 0, 1])

        action_index = int(np.argmax(self.q_table[state_id]))
        return [-1, 0, 1][action_index]

    def get_state_id(self, rel_pos):
        if isinstance(rel_pos, list):
            rel_pos = rel_pos[-1]
        return f"bb_{int(float(rel_pos) * 100)}"

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

    def calculate_bollinger_bands(self, close_prices):
        rolling_mean = close_prices.rolling(window=self.window).mean()
        rolling_std = close_prices.rolling(window=self.window).std()
        upper_band = rolling_mean + (rolling_std * self.num_std_dev)
        lower_band = rolling_mean - (rolling_std * self.num_std_dev)
        return upper_band, lower_band

    def save_model(self, filepath):
        model_data = {
            "window": self.window,
            "num_std_dev": self.num_std_dev,
            "q_table": self.q_table
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.window = model_data["window"]
        self.num_std_dev = model_data["num_std_dev"]
        self.q_table = model_data.get("q_table", {})
        self.required_window = self.window