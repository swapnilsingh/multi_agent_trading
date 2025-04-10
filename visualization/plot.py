# visualization/plot.py
import matplotlib.pyplot as plt

def plot_strategy(df, trades, capital_curve):
    plt.figure(figsize=(12, 6))
    prices = df['close'].values
    times = df.index[:len(prices)]

    plt.plot(times, prices, label='Price', color='gray')

    buy_points = [price for (_, action, price) in trades if action == 1]
    sell_points = [price for (_, action, price) in trades if action == -1]

    if buy_points:
        plt.scatter(times[:len(buy_points)], buy_points, color='green', marker='^', label='Buy')
    if sell_points:
        plt.scatter(times[:len(sell_points)], sell_points, color='red', marker='v', label='Sell')

    plt.title("Strategy Trades")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
