import pandas as pd
import matplotlib.pyplot as plt

def plot_equity_curve(balance_history):
    plt.figure()
    plt.plot(balance_history, label="Equity Curve", linewidth=2)
    plt.title("Equity Curve")
    plt.xlabel("Step")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_trades(prices, trades):
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label="Price", linewidth=1.5)

    for i, (t_type, price, _) in enumerate(trades):
        color = "green" if t_type == "BUY" else "red"
        plt.scatter(i, price, color=color, marker="^" if t_type == "BUY" else "v", label=t_type if i == 0 else "", zorder=5)

    plt.title("Trade Points")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
