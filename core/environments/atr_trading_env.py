# core/environments/atr_trading_env.py
import numpy as np
from core.environments.base_trading_environment import BaseTradingEnv

class ATRTradingEnv(BaseTradingEnv):
    def __init__(self, df, atr_period=14, **kwargs):
        super().__init__(df, **kwargs)
        self.atr_period = atr_period

    def _get_state(self):
        """
        Returns a dict containing historical high, low, and close data
        used by ATRAgent to calculate ATR.
        """
        window_data = self.df.iloc[self.current_step - self.window_size : self.current_step]
        state = {
            "high_history": window_data["high"].tolist(),
            "low_history": window_data["low"].tolist(),
            "close_history": window_data["close"].tolist(),
        }
        return state

    def execute_action(self, action):
        """
        Standard buy/sell/hold execution and reward computation.
        """
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0

        if action == 1:  # BUY
            if self.position == 0:
                self.entry_price = current_price
                self.position = 1
            elif self.position == -1:
                reward = self.entry_price - current_price
                self.balance += reward
                self.position = 0
                self.entry_price = None

        elif action == -1:  # SELL
            if self.position == 0:
                self.entry_price = current_price
                self.position = -1
            elif self.position == 1:
                reward = current_price - self.entry_price
                self.balance += reward
                self.position = 0
                self.entry_price = None

        self.current_step += 1
        done = self.current_step >= len(self.df)
        return reward, done
