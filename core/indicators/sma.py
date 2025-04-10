import pandas as pd

def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        series (pd.Series): The data series to average (e.g., closing prices)
        window (int): The rolling window size

    Returns:
        pd.Series: The SMA values
    """
    return series.rolling(window=window).mean()
