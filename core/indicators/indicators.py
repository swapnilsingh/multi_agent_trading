# indicators.py
import pandas as pd
import pandas_ta as ta

def compute_macd(data):
    if len(data) < 35:  # safe default for MACD(12,26,9)
        return data, 0.0

    try:
        macd_df = ta.macd(data['close'])

        if macd_df is None or not isinstance(macd_df, pd.DataFrame):
            return data, 0.0

        if macd_df.isnull().all().all():
            return data, 0.0

        required_cols = ['MACD_12_26_9', 'MACDs_12_26_9']
        if not all(col in macd_df.columns for col in required_cols):
            return data, 0.0

        # Drop existing if re-calculating
        data = data.drop(columns=[col for col in ['macd', 'macds'] if col in data.columns], errors='ignore')

        # Join MACD columns and rename them
        macd_df = macd_df.rename(columns={
            'MACD_12_26_9': 'macd',
            'MACDs_12_26_9': 'macd_signal'
        })

        data = data.join(macd_df)

        macd_val = data['macd'].iloc[-1]
        signal_val = data['macds'].iloc[-1]

        if pd.isna(macd_val) or pd.isna(signal_val):
            return data, 0.0

        return data, macd_val - signal_val

    except Exception as e:
        print(f"[MACD ERROR]: {e}")
        return data, 0.0

def compute_rsi(data, length=14):
    rsi_series = ta.rsi(data['close'], length=length)
    data = data.copy()
    data['rsi'] = rsi_series
    return data, rsi_series.fillna(0).iloc[-1]

def compute_bollinger(data, length=20, std=2):
    bb_df = ta.bbands(data['close'], length=length, std=std)
    data = data.drop(columns=[col for col in bb_df.columns if col in data.columns], errors='ignore')
    data = data.join(bb_df)
    price = data['close'].iloc[-1]
    upper = data['BBU_20_2.0'].fillna(0).iloc[-1]
    lower = data['BBL_20_2.0'].fillna(0).iloc[-1]
    return data, (price, upper, lower)

def compute_atr(data, length=14):
    atr_series = ta.atr(high=data['high'], low=data['low'], close=data['close'], length=length)
    data = data.copy()
    data['atr'] = atr_series
    return data, atr_series.fillna(0).iloc[-1]
