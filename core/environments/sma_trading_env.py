from core.environments.base_trading_environment import BaseTradingEnv
import numpy as np

class SMATradingEnv(BaseTradingEnv):
    def __init__(self, df, short_window=10, long_window=30, **kwargs):
        super().__init__(df, **kwargs)
        self.short_window = short_window
        self.long_window = long_window

    def _get_state(self):
        """
        Overriding the base method to include SMA values in the state.
        :return: A dictionary containing close, high, low prices, and the SMAs
        """
        state = super()._get_state()
        
        # Calculate short and long SMAs
        close_prices = state['close_history']
        short_sma = np.mean(close_prices[-self.short_window:])
        long_sma = np.mean(close_prices[-self.long_window:])
        
        state['short_sma'] = short_sma
        state['long_sma'] = long_sma
        
        return state

    def execute_action(self, action):
        """
        Overriding the base method to add the logic for SMA strategy.
        """
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0

        # If action is buy (1), sell (-1), or hold (0), execute the trade
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
