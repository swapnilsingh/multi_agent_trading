# core/agents/bollinger_agent.py
import pandas as pd
import numpy as np
import joblib
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager

class BollingerAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config, model_manager)
        self.window = config['bb_period']
        self.std_factor = config['bb_devfactor']
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 0.2)
        self.model_manager = model_manager

        self.required_window = self.window
        self.q_table = {}  # state_id -> [q_sell, q_hold, q_buy]
        self.state_action_counter = {}  # (state_id, action) -> count

    def act(self, state):
        close_history = pd.Series(state['close_history'])

        if not self.has_sufficient_data(close_history):
            return None

        bb_upper, bb_lower = self.calculate_bollinger_bands(close_history)
        price = close_history.iloc[-1]

        state_id = self.get_state_id({"price": price, "bb_upper": bb_upper, "bb_lower": bb_lower})

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]  # sell, hold, buy

        if np.random.rand() < self.epsilon:
            action = np.random.choice([-1, 0, 1])
        else:
            action_index = int(np.argmax(self.q_table[state_id]))
            action = [-1, 0, 1][action_index]

        key = (state_id, action)
        self.state_action_counter[key] = self.state_action_counter.get(key, 0) + 1

        return action

    def get_state_id(self, state):
        price = float(state.get("price", 0))
        upper = float(state.get("bb_upper", 0))
        lower = float(state.get("bb_lower", 0))

        if price > upper:
            return "above_upper"
        elif price < lower:
            return "below_lower"
        else:
            return "within_band"

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


    def calculate_bollinger_bands(self, close_prices):
        rolling_mean = close_prices.rolling(window=self.window).mean()
        rolling_std = close_prices.rolling(window=self.window).std()
        upper_band = rolling_mean.iloc[-1] + self.std_factor * rolling_std.iloc[-1]
        lower_band = rolling_mean.iloc[-1] - self.std_factor * rolling_std.iloc[-1]
        return upper_band, lower_band

    def save_model(self, filepath):
        joblib.dump({
            "bb_period": self.window,
            "bb_devfactor": self.std_factor,
            "q_table": self.q_table
        }, filepath)

    def load_model(self, filepath):
        data = joblib.load(filepath)
        self.window = data["bb_period"]
        self.std_factor = data["bb_devfactor"]
        self.q_table = data.get("q_table", {})
        self.required_window = self.window
