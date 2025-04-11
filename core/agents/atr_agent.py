# core/agents/atr_agent.py
import pandas as pd
import numpy as np
import joblib
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager

class ATRAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config)
        self.window = config['window']
        self.atr_threshold = config['atr_threshold']
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 0.2)
        self.model_manager = model_manager

        self.required_window = self.window
        self.q_table = {}  # state_id -> [q_sell, q_hold, q_buy]

    def act(self, state):
        high = pd.Series(state['high_history'])
        low = pd.Series(state['low_history'])
        close = pd.Series(state['close_history'])

        if not self.has_sufficient_data(close):
            return None

        atr = self.calculate_atr(high, low, close)
        if len(atr) == 0:
            return None

        current_atr = atr.iloc[-1]
        state_id = self.get_state_id(current_atr)

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if np.random.rand() < self.epsilon:
            return np.random.choice([-1, 0, 1])

        action_index = int(np.argmax(self.q_table[state_id]))
        return [-1, 0, 1][action_index]

    def get_state_id(self, atr_value):
        if isinstance(atr_value, list):
            atr_value = atr_value[-1]
        return f"atr_{int(float(atr_value) * 100)}"

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

    def calculate_atr(self, high, low, close):
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (close.shift() - low).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(window=self.window).mean()

    def save_model(self, filepath):
        joblib.dump({
            "window": self.window,
            "atr_threshold": self.atr_threshold,
            "q_table": self.q_table
        }, filepath)

    def load_model(self, filepath):
        data = joblib.load(filepath)
        self.window = data["window"]
        self.atr_threshold = data["atr_threshold"]
        self.q_table = data.get("q_table", {})
        self.required_window = self.window