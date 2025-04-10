from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def generate_signal(self, data: pd.Series) -> int:
        """
        Generate a trading signal based on the latest market data.

        Args:
            data (pd.Series): A single row of market data (e.g., OHLCV + indicators)

        Returns:
            int: -1 = sell, 0 = hold, 1 = buy
        """
        pass

    def generate_batch_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate a series of signals for an entire DataFrame of historical data.

        Args:
            df (pd.DataFrame): Historical market data

        Returns:
            pd.Series: Trading signals for each row
        """
        return df.apply(self.generate_signal, axis=1)
