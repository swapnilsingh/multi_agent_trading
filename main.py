import pandas as pd
from core.environments.trading_environment import TradingEnvironment
from core.utils.config_loader import load_config
from core.utils.logger import get_logger

# --- Import your agent and strategy here ---
from core.agents.rsi_agent import RSIAgent
from core.strategies.rsi_strategy import RSIStrategy

def run():
    # Load market data (example)
    data = pd.read_csv("data/processed/btc_ohlcv.csv")

    # Load config
    config = load_config("configs/strategies/rsi.yaml")

    # Init components
    logger = get_logger("trading-session")
    env = TradingEnvironment(market_data=data, initial_balance=10_000)
    strategy = RSIStrategy(config)
    agent = RSIAgent(config)

    # Run simulation
    state = env.reset()
    done = False

    while not done:
        # Package state for strategy (example: wrap the close history)
        step = env.current_step
        close_window = data["close"].iloc[:step+1]
        strategy_input = {
            "close_history": close_window
        }

        action = strategy.generate_signal(strategy_input)
        logger.info(f"Step {step} | Action: {action}")

        state, reward, done = env.step(action)

    logger.info(f"Final Balance: {env.balance}")
    print("Trades executed:", env.trades)

if __name__ == "__main__":
    run()
