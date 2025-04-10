import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from core.environments.trading_environment import TradingEnvironment
from core.utils.logger import get_logger
from core.utils.config_loader import load_config

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

# Path to CSV
CSV_PATH = "data/processed/btc_ohlcv.csv"

# Test each agent-strategy pair
TESTS = [
    ("RSI", RSIAgent, RSIStrategy, "configs/strategies/rsi.yaml"),
    ("MACD", MACDAgent, MACDStrategy, "configs/strategies/macd.yaml"),
    ("Bollinger", BollingerAgent, BollingerStrategy, "configs/strategies/bollinger.yaml"),
    ("ATR", ATRAgent, ATRStrategy, "configs/strategies/atr.yaml"),
    ("SMA", SMAAgent, SMAStrategy, "configs/strategies/sma.yaml"),
]

def run_test(name, AgentClass, StrategyClass, config_path):
    print(f"\n🔍 Testing {name} Agent & Strategy")

    data = pd.read_csv(CSV_PATH)
    config = load_config(config_path)

    env = TradingEnvironment(market_data=data, initial_balance=10_000)
    agent = AgentClass(config)
    strategy = StrategyClass(config)

    state = env.reset()
    done = False

    while not done:
        step = env.current_step
        if step >= len(data):
            break

        # Build input state
        state_data = {
            "close_history": data["close"].iloc[:step+1],
            "high_history": data["high"].iloc[:step+1],
            "low_history": data["low"].iloc[:step+1]
        }

        action = strategy.generate_signal(state_data)
        state, reward, done = env.step(action)

    print(f"✅ Final Balance: {env.balance}")
    print(f"📊 Trades Executed: {len(env.trades)}")

if __name__ == "__main__":
    for name, Agent, Strategy, config in TESTS:
        run_test(name, Agent, Strategy, config)
