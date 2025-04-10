import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from indicators import compute_macd  # Assuming you have this

class MACDTradingEnv(gym.Env):
    def __init__(self, df, window_size=30, initial_balance=1000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Actions: Sell (-1), Hold (0), Buy (+1) → use index offset
        self.action_space = spaces.Discrete(3)

        # Observation: MACD line and signal line
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.inventory = 0
        self.net_worth = self.initial_balance
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.loc[self.current_step]
        return np.array([row['macd'], row['macds']], dtype=np.float32)

    def step(self, action_idx):
        action = action_idx - 1  # [0,1,2] → [-1, 0, 1]
        current_price = self.df.loc[self.current_step, 'close']
        reward = 0

        if action == 1 and self.balance >= current_price:
            self.inventory += 1
            self.balance -= current_price
        elif action == -1 and self.inventory > 0:
            self.inventory -= 1
            self.balance += current_price

        self.net_worth = self.balance + self.inventory * current_price
        reward = self.net_worth - self.initial_balance

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Inventory: {self.inventory}, Net Worth: {self.net_worth}")
