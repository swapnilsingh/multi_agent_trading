from core.environments.base_trading_environment import BaseTradingEnv
import numpy as np
import pandas as pd

class MACDTradingEnv(BaseTradingEnv):
    def __init__(self, df, fast_period=12, slow_period=26, signal_period=9, **kwargs):
        super().__init__(df, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def _get_state(self):
        """
        Overriding the base method to include the MACD and Signal Line values in the state.
        :return: A dictionary containing close, high, low prices, and MACD indicators
        """
        state = super()._get_state()
        
        # Calculate MACD and Signal Line
        close_prices = state['close_history']
        macd, signal_line = self._calculate_macd(close_prices)
        
        state['macd'] = macd
        state['signal_line'] = signal_line
        return state

    def _calculate_macd(self, prices):
        short_ema = self._calculate_ema(prices, self.fast_period)
        long_ema = self._calculate_ema(prices, self.slow_period)
        macd = short_ema - long_ema
        signal_line = self._calculate_ema(macd, self.signal_period)
        return macd.iloc[-1], signal_line.iloc[-1]


    def _calculate_ema(self, prices, period):
        """
        Calculate the Exponential Moving Average (EMA) for a given list of prices and period.
        :param prices: List of historical prices
        :param period: Period for EMA calculation
        :return: EMA values
        """
        return pd.Series(prices).ewm(span=period, adjust=False).mean()

    def execute_action(self, action):
        """
        Overriding the base method to add the logic for MACD strategy.
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
