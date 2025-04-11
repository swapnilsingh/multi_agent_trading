import pandas as pd
import numpy as np

class BaseTradingEnv:
    def __init__(self, df: pd.DataFrame, window_size=50, initial_balance=1000):
        """
        Initializes the trading environment with given data, window size, and initial balance.
        
        :param df: DataFrame containing OHLCV data
        :param window_size: The size of the window (number of past data points) used to represent the state
        :param initial_balance: The initial balance of the trading account
        """
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        
    def reset(self):
        """
        Resets the environment to the initial state.
        
        :return: Initial state of the environment.
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # +1 for long, -1 for short, 0 for neutral
        self.entry_price = None
        return self._get_state()

    def _get_state(self):
        """
        Returns the current state of the environment based on the current step.

        :return: A dictionary containing the historical close, high, and low prices
        """
        window = self.df.iloc[self.current_step - self.window_size : self.current_step]
        return {
            'close_history': window['Close'].tolist(),
            'high_history': window['High'].tolist(),
            'low_history': window['Low'].tolist(),
        }
    
    def get_current_state(self):
        return self._get_state()

    def execute_action(self, action):
        """
        Executes an action (buy, sell, or hold) and returns the reward and whether the episode is done.

        :param action: Action to be executed (1 for buy, -1 for sell, 0 for hold)
        :return: A tuple containing the reward and whether the episode is finished.
        """
        current_price = self.df.iloc[self.current_step]['Close']
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

    def skip(self):
        """
        Skips to the next time step without executing any action.
        """
        self.current_step += 1