import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df, window_size=30, initial_balance=1000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Actions: Discrete mapped to -1 (Sell), 0 (Hold), 1 (Buy)
        self.action_space = spaces.Discrete(3)

        # Observations: RSI value
        self.observation_space = spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.inventory = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        rsi = self.df.loc[self.current_step, 'rsi']
        return np.array([rsi], dtype=np.float32)

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        reward = 0

        # Execute action
        if action == 1:  # Buy
            if self.balance >= current_price:
                self.inventory += 1
                self.balance -= current_price
            else:
                reward -= 0.2  # Penalty for invalid buy

        elif action == -1:  # Sell
            if self.inventory > 0:
                self.inventory -= 1
                self.balance += current_price
            else:
                reward -= 0.2  # Penalty for invalid sell

        # Update net worth
        self.net_worth = self.balance + self.inventory * current_price

        # Reward: change in net worth
        reward += self.net_worth - self.prev_net_worth

        # Directional reward
        price_diff = current_price - self.df.loc[self.current_step - 1, 'close']
        if action == 1 and price_diff > 0:
            reward += 0.2  # Good buy
        elif action == -1 and price_diff < 0:
            reward += 0.2  # Good sell

        # Penalize frequent trading
        if action != 0:
            reward -= 0.1

        self.prev_net_worth = self.net_worth
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Inventory: {self.inventory}, Net Worth: {self.net_worth}")
