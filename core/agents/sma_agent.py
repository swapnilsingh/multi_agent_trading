# core/agents/sma_agent.py
import pandas as pd
import joblib
import numpy as np
from core.agents.base_agent import BaseAgent
from core.model_management.model_manager import ModelManager

class SMAAgent(BaseAgent):
    def __init__(self, config, model_manager: ModelManager):
        super().__init__(config, model_manager)
        self.short_window = config.get("short_window", 10)
        self.long_window = config.get("long_window", 30)
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 0.2)
        self.required_window = self.long_window
        self.q_table = {}
        self.model_manager = model_manager

    def act(self, state):
        close_prices = pd.Series(state.get("close_history", []))
        if len(close_prices) < self.required_window:
            return 0  # hold if not enough data

        sma_short = close_prices[-self.short_window:].mean()
        sma_long = close_prices[-self.long_window:].mean()

        state_id = self.get_state_id({"sma_short": sma_short, "sma_long": sma_long})

        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if np.random.rand() < self.epsilon:
            return np.random.choice([-1, 0, 1])

        action_index = int(np.argmax(self.q_table[state_id]))
        return [-1, 0, 1][action_index]

    def get_state_id(self, state):
        try:
            sma_short = state.get("sma_short", 0)
            sma_long = state.get("sma_long", 0)
            delta = sma_short - sma_long
            return f"sma_{int(delta * 100)}"
        except Exception as e:
            print(f"[{self.name}] Error computing state_id: {e}")
            return "sma_error"

    def learn(self, state_id, action, reward, next_state_id):
        if state_id not in self.q_table:
            self.q_table[state_id] = [0, 0, 0]

        if next_state_id not in self.q_table:
            self.q_table[next_state_id] = [0, 0, 0]

        action_index = action + 1  # -1=>0, 0=>1, 1=>2
        old_q = self.q_table[state_id][action_index]
        next_max_q = max(self.q_table[next_state_id])
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[state_id][action_index] = new_q

    def save_model(self, filepath):
        model_data = {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "q_table": self.q_table
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath):
        model_data = joblib.load(filepath)
        self.short_window = model_data.get("short_window", 10)
        self.long_window = model_data.get("long_window", 30)
        self.q_table = model_data.get("q_table", {})
