import pandas as pd
import matplotlib.pyplot as plt
from core.environments.trading_environment import TradingEnvironment
from core.utils.config_loader import load_config
from core.utils.logger import get_logger

# Agent + Strategy imports
from core.agents.rsi_agent import RSIAgent
from core.strategies.rsi_strategy import RSIStrategy

from core.agents.macd_agent import MACDAgent
from core.strategies.macd_strategy import MACDStrategy

from core.agents.bollinger_agent import BollingerAgent
from core.strategies.bollinger_strategy import BollingerStrategy

from core.agents.atr_agent import ATRAgent
from core.strategies.atr_strategy import ATRStrategy

from core.agents.sma_agent import SMAAgent
from core.strategies.sma_strategy import SMAStrategy


TESTS = [
    ("RSI", RSIAgent, RSIStrategy, "configs/strategies/rsi.yaml"),
    ("MACD", MACDAgent, MACDStrategy, "configs/strategies/macd.yaml"),
    ("Bollinger", BollingerAgent, BollingerStrategy, "configs/strategies/bollinger.yaml"),
    ("ATR", ATRAgent, ATRStrategy, "configs/strategies/atr.yaml"),
    ("SMA", SMAAgent, SMAStrategy, "configs/strategies/sma.yaml"),
]


def plot_equity_curve(balances, title):
    plt.figure()
    plt.plot(balances, label="Equity Curve", linewidth=2)
    plt.title(f"{title} - Equity Curve")
    plt.xlabel("Step")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_trades(prices, trades, title):
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label="Price", linewidth=1.5)

    for i, (t_type, price, _) in enumerate(trades):
        color = "green" if t_type in ("BUY", "COVER") else "red"
        marker = "^" if t_type in ("BUY", "COVER") else "v"
        plt.scatter(i, price, color=color, marker=marker, label=t_type if i == 0 else "", zorder=5)

    plt.title(f"{title} - Trades on Price")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run():
    data = pd.read_csv("data/processed/btc_ohlcv.csv")
    logger = get_logger("multi-agent-run", log_file="logs/multi_agent_run.log")

    for name, AgentClass, StrategyClass, config_path in TESTS:
        print(f"\n🔍 Running {name} Strategy")
        logger.info(f"--- Starting run for {name} ---")

        config = load_config(config_path)
        env = TradingEnvironment(market_data=data, initial_balance=10_000)
        agent = AgentClass(config)
        strategy = StrategyClass(config)

        state = env.reset()
        done = False
        balance_history = [env.balance]

        while not done:
            step = env.current_step
            if step >= len(data):
                break

            strategy_input = {
                "close_history": data["close"].iloc[:step+1],
                "high_history": data["high"].iloc[:step+1],
                "low_history": data["low"].iloc[:step+1]
            }

            action = strategy.generate_signal(strategy_input)
            logger.info(f"{name} | Step {step} | Action: {action}")
            state, reward, done = env.step(action)
            balance_history.append(env.balance)

        print(f"✅ {name} Final Balance: {env.balance:.2f}")
        print(f"📊 Trades Executed ({len(env.trades)}):")
        for trade in env.trades:
            print(f" - {trade}")

        logger.info(f"--- Completed run for {name}, Balance: {env.balance} ---\n")

        # 🔍 Visualizations
        plot_equity_curve(balance_history, name)
        plot_trades(data["close"], env.trades, name)


if __name__ == "__main__":
    run()
