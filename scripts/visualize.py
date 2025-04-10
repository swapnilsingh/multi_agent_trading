import matplotlib.pyplot as plt
import pandas_ta as ta


def plot_trades(data, trades):
    prices = data['close'].values
    times = data.index

    plt.figure(figsize=(14, 6))
    plt.plot(times, prices, label='Price', color='blue')

    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for i, (action, price) in enumerate(trades):
        if action == 'buy':
            buy_x.append(times[i])
            buy_y.append(price)
        elif action == 'sell':
            sell_x.append(times[i])
            sell_y.append(price)

    plt.scatter(buy_x, buy_y, color='green', label='Buy', marker='^')
    plt.scatter(sell_x, sell_y, color='red', label='Sell', marker='v')

    plt.title("Trade Signals on Price Chart")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_capital_evolution(trades, starting_capital):
    capital = starting_capital
    capital_history = [capital]

    for action, price in trades:
        if action == 'buy':
            capital -= price
        elif action == 'sell':
            capital += price
        capital_history.append(capital)

    plt.figure(figsize=(10, 4))
    plt.plot(capital_history, label='Capital Over Time', color='orange')
    plt.title("Capital Evolution")
    plt.xlabel("Trade #")
    plt.ylabel("Capital")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_rsi(data):
    rsi = ta.rsi(data['close'], length=14)
    if rsi is None or rsi.isnull().all():
        print("[RSI] Not enough data.")
        return

    plt.figure(figsize=(12, 3))
    plt.plot(data.index, rsi, label='RSI (14)', color='purple')
    plt.axhline(70, linestyle='--', color='red', alpha=0.5)
    plt.axhline(30, linestyle='--', color='green', alpha=0.5)
    plt.title("Relative Strength Index (RSI)")
    plt.ylabel("RSI")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_macd(data):
    macd_df = ta.macd(data['close'])
    if macd_df is None or macd_df.isnull().all().all():
        print("[MACD] Not enough data.")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(data.index, macd_df['MACD_12_26_9'], label='MACD Line', color='blue')
    plt.plot(data.index, macd_df['MACDs_12_26_9'], label='Signal Line', color='orange')
    plt.bar(data.index, macd_df['MACDh_12_26_9'], label='Histogram', color='gray', alpha=0.3)
    plt.title("MACD (12, 26, 9)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bollinger(data):
    bb = ta.bbands(data['close'], length=20)
    if bb is None or bb.isnull().all().all():
        print("[Bollinger] Not enough data.")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(data.index, data['close'], label='Close', color='blue')
    plt.plot(data.index, bb['BBL_20_2.0'], label='Lower Band', linestyle='--', color='green')
    plt.plot(data.index, bb['BBM_20_2.0'], label='Middle Band', linestyle=':', color='gray')
    plt.plot(data.index, bb['BBU_20_2.0'], label='Upper Band', linestyle='--', color='red')
    plt.fill_between(data.index, bb['BBL_20_2.0'], bb['BBU_20_2.0'], color='lightgray', alpha=0.2)
    plt.title("Bollinger Bands (20, 2.0)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_atr(data):
    atr = ta.atr(data['high'], data['low'], data['close'], length=14)
    if atr is None or atr.isnull().all():
        print("[ATR] Not enough data.")
        return

    plt.figure(figsize=(12, 3))
    plt.plot(data.index, atr, label='ATR (14)', color='brown')
    plt.title("Average True Range (ATR)")
    plt.ylabel("ATR")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
