import numpy as np
import pandas as pd
import joblib
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager
from core.indicators.atr import calculate_atr

class ATRAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config, model_manager)
        self.atr_period = config['atr_period']
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 0.2)
        self.model_manager = model_manager

        self.required_window = self.atr_period + 1
        self.q_table = {}
        self.state_action_counter = {}

    def act(self, state):
        try:
            high = state.get('high_history', [])
            low = state.get('low_history', [])
            close = state.get('close_history', [])

            if not self.has_sufficient_data(high) or not self.has_sufficient_data(low) or not self.has_sufficient_data(close):
                return 0

            atr = self.calculate_atr(high, low, close)
            state_id = self.get_state_id(atr)

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

        except Exception as e:
            print(f"[{self.name}] ⚠️ Error during training step: {e}")
            return 0

    def get_state_id(self, atr_value):
        try:
            bucket = int(float(atr_value) // 5) * 5  # Bucket to nearest lower multiple of 5
            return f"atr_{bucket}"
        except Exception as e:
            print(f"[{self.name}] Error computing state_id: {e}")
            return "atr_error"

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

    def calculate_atr(self, high_prices, low_prices, close_prices):
        if not self.has_sufficient_data(high_prices) or not self.has_sufficient_data(low_prices) or not self.has_sufficient_data(close_prices):
            return 0.0
        high_series = pd.Series(high_prices)
        low_series = pd.Series(low_prices)
        close_series = pd.Series(close_prices)
        atr_values = calculate_atr(high_series, low_series, close_series, window=self.atr_period)
        atr_value = atr_values.iloc[-1]
        if pd.isna(atr_value):
            return 0.0
        return float(atr_value)

    def save_model(self, filepath):
        joblib.dump({
            "atr_period": self.atr_period,
            "q_table": self.q_table
        }, filepath)

    def load_model(self, filepath):
        data = joblib.load(filepath)
        self.atr_period = data["atr_period"]
        self.q_table = data.get("q_table", {})
        self.required_window = self.atr_period + 1
