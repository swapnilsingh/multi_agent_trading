# core/agents/rsi_agent.py
import numpy as np
import pandas as pd
import joblib
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager

class RSIAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config)
        self.rsi_period = config['rsi_period']
        self.required_window = self.rsi_period
        self.model_manager = model_manager

        # Q-learning setup
        self.q_table = {}  # state_id: [q_sell, q_hold, q_buy]
        self.alpha = config.get('alpha', 0.1)
        self.gamma = config.get('gamma', 0.95)
        self.epsilon = config.get('epsilon', 0.2)

    def act(self, state):
        close_history = pd.Series(state['close_history'])

        if not self.has_sufficient_data(close_history):
            print(f"[RSIAgent] Not enough data. Required: {self.required_window}, Got: {len(close_history)}")
            return None

        state_id = self.get_state_id(close_history)

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]  # [q_sell, q_hold, q_buy]

        if np.random.rand() < self.epsilon:
            return np.random.choice([-1, 0, 1])  # Explore

        action_index = np.argmax(self.q_table[state_id])
        return [-1, 0, 1][action_index]  # Exploit

    def train(self, experience):
        state_id, action, reward, next_state_id = experience

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if next_state_id not in self.q_table:
            self.q_table[next_state_id] = [0, 0, 0]

        action_index = action + 1  # -1 -> 0, 0 -> 1, 1 -> 2

        old_q = self.q_table[state_id][action_index]
        next_max_q = max(self.q_table[next_state_id])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)

        self.q_table[state_id][action_index] = new_q

    def get_state_id(self, close_history):
        close_series = pd.Series(close_history)
        rsi = self.calculate_rsi(close_series)
        last_rsi = int(rsi.iloc[-1]) if not rsi.isna().all() else 50
        return str(last_rsi // 10)  # Bucketed RSI like '4', '5', '6'


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
            "rsi_period": self.rsi_period,
            "q_table": self.q_table
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.rsi_period = model_data["rsi_period"]
        self.q_table = model_data.get("q_table", {})
        self.required_window = self.rsi_period
