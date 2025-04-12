from core.environments.base_trading_environment import BaseTradingEnv
import numpy as np
import pandas as pd

class BollingerTradingEnv(BaseTradingEnv):
    def __init__(self, df, window_size=20, num_std=2, **kwargs):
        super().__init__(df, **kwargs)
        self.window_size = window_size
        self.num_std = num_std

    def _get_state(self):
        """
        Overriding the base method to include the Bollinger Bands values in the state.
        :return: A dictionary containing close, high, low prices, and Bollinger Bands indicators
        """
        state = super()._get_state()
        
        # Calculate Bollinger Bands
        close_prices = state['close_history']
        upper_band, lower_band, moving_avg = self._calculate_bollinger_bands(close_prices)
        
        state['upper_band'] = upper_band
        state['lower_band'] = lower_band
        state['moving_avg'] = moving_avg
        return state

    def _calculate_bollinger_bands(self, prices):
        """
        Calculate the upper and lower Bollinger Bands for a given list of prices.
        :param prices: List of historical close prices
        :return: upper_band, lower_band, moving_avg
        """
        close_series = pd.Series(prices)
        moving_avg = close_series.rolling(window=self.window_size).mean()
        rolling_std = close_series.rolling(window=self.window_size).std()
        
        upper_band = moving_avg + (self.num_std * rolling_std)
        lower_band = moving_avg - (self.num_std * rolling_std)

        return upper_band.iloc[-1], lower_band.iloc[-1], moving_avg.iloc[-1]

    def execute_action(self, action):
        """
        Overriding the base method to add the logic for Bollinger Bands strategy.
        """
        current_price = self.df.iloc[self.current_step]['close']
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
