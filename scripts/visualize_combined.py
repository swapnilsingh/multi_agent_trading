import matplotlib.pyplot as plt
import pandas_ta as ta

def plot_combined(data, trades, capital_start):
    macd = ta.macd(data['close'])
    rsi = ta.rsi(data['close'], length=14)
    bb = ta.bbands(data['close'], length=20)
    atr = ta.atr(data['high'], data['low'], data['close'], length=14)

    # Compute capital
    capital = capital_start
    capital_history = [capital]
    for action, price in trades:
        if action == 'buy':
            capital -= price
        elif action == 'sell':
            capital += price
        capital_history.append(capital)

    fig, axs = plt.subplots(5, 1, figsize=(15, 12), sharex=True,
                            gridspec_kw={'height_ratios': [3, 1, 1, 1, 2]})
    times = data.index
    price = data['close']

    # ----- PRICE + TRADES + BOLLINGER -----
    axs[0].plot(times, price, label='Price', color='blue')
    if bb is not None and not bb.isnull().all().all():
        axs[0].plot(times, bb['BBL_20_2.0'], linestyle='--', color='green', label='BB Lower')
        axs[0].plot(times, bb['BBU_20_2.0'], linestyle='--', color='red', label='BB Upper')
    for i, (action, p) in enumerate(trades):
        if action == 'buy':
            axs[0].scatter(times[i], p, color='green', marker='^')
        elif action == 'sell':
            axs[0].scatter(times[i], p, color='red', marker='v')
    axs[0].set_title('Price + Trade Signals + Bollinger')
    axs[0].legend()
    axs[0].grid()

    # ----- RSI -----
    axs[1].plot(times, rsi, label='RSI (14)', color='purple')
    axs[1].axhline(70, linestyle='--', color='red', alpha=0.5)
    axs[1].axhline(30, linestyle='--', color='green', alpha=0.5)
    axs[1].set_title('RSI')
    axs[1].legend()
    axs[1].grid()

    # ----- MACD -----
    if macd is not None and not macd.isnull().all().all():
        axs[2].plot(times, macd['MACD_12_26_9'], label='MACD', color='blue')
        axs[2].plot(times, macd['MACDs_12_26_9'], label='Signal', color='orange')
        axs[2].bar(times, macd['MACDh_12_26_9'], label='Hist', color='gray', alpha=0.3)
    axs[2].set_title('MACD')
    axs[2].legend()
    axs[2].grid()

    # ----- ATR -----
    axs[3].plot(times, atr, label='ATR (14)', color='brown')
    axs[3].set_title('ATR')
    axs[3].legend()
    axs[3].grid()

    # ----- CAPITAL EVOLUTION -----
    axs[4].plot(capital_history, label='Capital', color='orange')
    axs[4].set_title('Capital Over Time')
    axs[4].legend()
    axs[4].grid()

    plt.tight_layout()
    plt.show()
