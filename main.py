import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import json
import csv
import zipfile
import os

from indicators import compute_rsi, compute_macd, compute_atr, compute_bollinger
from data.market_data import fetch_market_data
from agents.rsi_agent import RSIAgent
from agents.macd_agent import MACDAgent
from agents.bollinger_agent import BollingerAgent
from core.aggregator import aggregate_actions

# === Load Data ===
df = fetch_market_data(symbol='BTC/USDT', timeframe='1m', limit=500)
df, _ = compute_rsi(df)
df, _ = compute_macd(df)
df, _ = compute_atr(df)
df, _ = compute_bollinger(df)  # ensure bb_upper and bb_lower are saved

# === Initialize Agents ===
class RSIAgent:
    def __init__(self, name="RSI", lower=30, upper=70):
        self.name = name
        self.lower = lower
        self.upper = upper

    def preprocess(self, data):
        if len(data) < 2:
            return 0

        prev_rsi = data.iloc[-2]['rsi']
        curr_rsi = data.iloc[-1]['rsi']

        if prev_rsi < self.lower and curr_rsi >= self.lower:
            print(f"RSIAgent: RSI crossed above {self.lower} → BUY")
            return 1
        elif prev_rsi > self.upper and curr_rsi <= self.upper:
            print(f"RSIAgent: RSI crossed below {self.upper} → SELL")
            return -1
        return 0

agents = {
    "RSI": RSIAgent(),
    "MACD": MACDAgent(),
    "Bollinger": BollingerAgent(),
}

# === Order Management System ===
class OrderManagementSystem:
    def __init__(self, initial_capital):
        self.capital = initial_capital
        self.inventory = 0
        self.trade_history = []
        self.equity_curve = []

    def execute(self, action, price):
        if action == 1 and self.capital >= price:
            self.inventory += 1
            self.capital -= price
            self.trade_history.append(('BUY', price))
            return True
        elif action == -1 and self.inventory > 0:
            self.inventory -= 1
            self.capital += price
            self.trade_history.append(('SELL', price))
            return True
        return False

    def update_equity(self, price):
        equity = self.capital + self.inventory * price
        self.equity_curve.append(equity)

    def summary(self):
        return {
            'Final Capital': self.capital + self.inventory * df.iloc[-1]['close'],
            'Inventory Left': self.inventory,
            'Total Trades': len(self.trade_history)
        }

# === Simulation Loop ===
oms = OrderManagementSystem(initial_capital=1000)
trade_log = []
action_counts = Counter()

for step in range(30, len(df)):
    sliced_data = df.iloc[:step+1].copy()
    price = sliced_data.iloc[-1]['close']

    votes = {name: agent.preprocess(sliced_data) for name, agent in agents.items()}
    final_action = aggregate_actions(votes.values())

    print(f"Step {step} Votes: {votes} → Final Action: {final_action}")

    trade_occurred = oms.execute(final_action, price)  # track if trade executed
    oms.update_equity(price)

    if trade_occurred:
        trade_log.append((step, final_action, price))

    action_counts[final_action] += 1

# === Summary ===
summary = oms.summary()
print("\n===== TRADE SUMMARY =====")
print(summary)
print("\n===== ACTION DISTRIBUTION =====")
print(action_counts)

# === Visualization ===
def plot_strategy(df, trade_log, capital_curve):
    buy_signals = [t[0] for t in trade_log if t[1] == 1]
    sell_signals = [t[0] for t in trade_log if t[1] == -1]
    prices = df['close'].values
    upper_band = df.get('BBU_20_2.0', pd.Series(index=df.index)).values
    lower_band = df.get('BBL_20_2.0', pd.Series(index=df.index)).values

    fig, axs = plt.subplots(4, 1, figsize=(16, 14), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    axs[0].plot(prices, label='Price', color='black')
    axs[0].plot(upper_band, label='BB Upper', linestyle='--')
    axs[0].plot(lower_band, label='BB Lower', linestyle='--')
    axs[0].scatter(buy_signals, prices[buy_signals], marker='^', color='green', label='Buy', zorder=5)
    axs[0].scatter(sell_signals, prices[sell_signals], marker='v', color='red', label='Sell', zorder=5)
    axs[0].set_title('Price with Buy/Sell and Bollinger Bands')
    axs[0].legend()

    axs[1].plot(df['rsi'].values, label='RSI', color='blue')
    axs[1].axhline(70, color='red', linestyle='--')
    axs[1].axhline(30, color='green', linestyle='--')
    axs[1].set_title('RSI')
    axs[1].legend()

    axs[2].plot(df['macd'].values, label='MACD', color='purple')
    axs[2].plot(df['macds'].values if 'macds' in df.columns else df['macd_signal'].values, label='Signal', linestyle='--', color='orange')
    axs[2].set_title('MACD')
    axs[2].legend()

    axs[3].plot(capital_curve, label='Equity Curve', color='teal')
    axs[3].set_title('Capital Over Time')
    axs[3].legend()

    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig("strategy_overview.png")
    plt.show()

plot_strategy(df, trade_log, oms.equity_curve)

# === Export Results ===
def export_results(summary, trade_log, capital_curve):
    os.makedirs("exports", exist_ok=True)

    with open("exports/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("exports/trade_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Action", "Price"])
        writer.writerows(trade_log)

    with open("exports/capital_curve.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Capital"])
        for i, val in enumerate(capital_curve):
            writer.writerow([i, val])

    with zipfile.ZipFile("exports/results.zip", "w") as zipf:
        zipf.write("exports/summary.json")
        zipf.write("exports/trade_log.csv")
        zipf.write("exports/capital_curve.csv")
        zipf.write("strategy_overview.png")

export_results(summary, trade_log, oms.equity_curve)
