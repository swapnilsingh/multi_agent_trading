# core/environments/rsi_trading_env.py
import numpy as np
import pandas as pd
from core.environments.base_trading_environment import BaseTradingEnv

class RSITRadingEnv(BaseTradingEnv):
    def __init__(self, df: pd.DataFrame, rsi_period=14, **kwargs):
        super().__init__(df, **kwargs)  # Call parent class constructor
        self.rsi_period = rsi_period  # Initialize rsi_period attribute
        self.reset()

    def _get_state(self):
        """
        Retrieve the current state, which includes RSI.
        """
        close_prices = self.df.iloc[self.current_step - self.window_size: self.current_step]['Close'].values
        rsi = self._calculate_rsi(close_prices)
        state = {
            "close_history": close_prices.tolist(),
            "rsi": rsi  # Add RSI to the statel
        }
        return state

    def _calculate_rsi(self, close_prices):
        """
        Calculate the Relative Strength Index (RSI) for the given close prices.
        """
        gain = np.diff(close_prices)
        loss = -gain[gain < 0]  # Negative gains
        gain = gain[gain > 0]  # Positive gains
        
        # Calculate the average gain and loss over the period
        avg_gain = np.mean(gain[-self.rsi_period:]) if len(gain) >= self.rsi_period else 0
        avg_loss = np.mean(loss[-self.rsi_period:]) if len(loss) >= self.rsi_period else 0

        # Handle case where avg_loss is 0
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
