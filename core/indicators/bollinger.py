import pandas as pd

def calculate_bollinger_bands(close: pd.Series, window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        close (pd.Series): Series of closing prices
        window (int): Rolling window size
        std_dev (float): Standard deviation multiplier

    Returns:
        pd.DataFrame: columns ['middle_band', 'upper_band', 'lower_band']
    """
    middle_band = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std()

    upper_band = middle_band + (std_dev * rolling_std)
    lower_band = middle_band - (std_dev * rolling_std)

    return pd.DataFrame({
        "middle_band": middle_band,
        "upper_band": upper_band,
        "lower_band": lower_band
    })
