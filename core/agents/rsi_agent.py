# core/agents/rsi_agent.py
import numpy as np
import pandas as pd
import joblib
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager

class RSIAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config, model_manager)
        self.rsi_period = config['rsi_period']
        self.required_window = self.rsi_period
        self.model_manager = model_manager

        self.q_table = {}  # state_id -> [q_sell, q_hold, q_buy]
        self.alpha = config.get('alpha', 0.1)
        self.gamma = config.get('gamma', 0.95)
        self.epsilon = config.get('epsilon', 0.2)
        self.state_action_counter = {}  # (state_id, action) -> count

    def act(self, state):
        close_history = state.get("close_history", [])

        # Flatten if needed
        if isinstance(close_history[0], list):
            close_history = [x[0] for x in close_history]

        close_series = pd.Series(close_history, dtype='float64')
        rsi = self.calculate_rsi(close_series)
        state_id = str(rsi // 10)

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if np.random.rand() < self.epsilon:
            action = np.random.choice([-1, 0, 1])
        else:
            action_index = int(np.argmax(self.q_table[state_id]))
            action = [-1, 0, 1][action_index]

        key = (state_id, action)
        self.state_action_counter[key] = self.state_action_counter.get(key, 0) + 1

        return action


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

    def get_state_id(self, state):
        close_history = state.get("close_history", [])
        if isinstance(close_history[0], list):
            close_history = [x[0] for x in close_history]

        close_series = pd.Series(close_history, dtype='float64')
        rsi = self.calculate_rsi(close_series)
        return str(rsi // 10)

    def calculate_rsi(self, close_series):
        delta = close_series.diff().fillna(0)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.rsi_period).mean().iloc[-1]
        avg_loss = loss.rolling(window=self.rsi_period).mean().iloc[-1]

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def save_model(self, filepath):
        joblib.dump({
            "rsi_period": self.rsi_period,
            "q_table": self.q_table
        }, filepath)

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.rsi_period = model_data["rsi_period"]
        self.q_table = model_data.get("q_table", {})
        self.required_window = self.rsi_period
