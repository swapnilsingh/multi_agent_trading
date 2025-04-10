# main_infer.py
import pandas as pd
from core.environments.multi_agent_trading_environment import MultiAgentTradingEnv
from core.agents.rsi_agent import RSIAgent
from core.agents.macd_agent import MACDAgent
from core.agents.bollinger_agent import BollingerAgent
from core.agents.atr_agent import ATRAgent
from core.agents.sma_agent import SMAAgent
from core.model_management.model_manager import ModelManager
from core.utils.config_loader import load_config
import os

# Load historical data
df = pd.read_csv("data/live_binance_ohlcv.csv")

# Initialize environment
env = MultiAgentTradingEnv(df, window_size=50, initial_balance=1000)

# Load configs from YAML
rsi_config = load_config("configs/strategies/rsi.yaml")
macd_config = load_config("configs/strategies/macd.yaml")
bollinger_config = load_config("configs/strategies/bollinger.yaml")
atr_config = load_config("configs/strategies/atr.yaml")
sma_config = load_config("configs/strategies/sma.yaml")

# Initialize ModelManager
model_manager = ModelManager()

# Instantiate agents
agents = [
    RSIAgent(rsi_config, model_manager),
    MACDAgent(macd_config, model_manager),
    BollingerAgent(bollinger_config, model_manager),
    ATRAgent(atr_config, model_manager),
    SMAAgent(sma_config, model_manager)
]

# Load pre-trained models
for agent in agents:
    model_path = f"models/{agent.__class__.__name__}.pkl"
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"[Model] Loaded: {model_path}")
    else:
        print(f"[Warning] No model found for {agent.__class__.__name__}, using default params.")

# Run inference
env.reset()
step = 0

while True:
    state = env.get_current_state()
    print(f"\n[Step {step}] Close Price: {state['close_history'][-1]}")

    for agent in agents:
        action = agent.act(state)
        action_str = "BUY" if action == 1 else "SELL" if action == -1 else "HOLD" if action == 0 else "WAIT"
        print(f"{agent.__class__.__name__}: {action_str}")

    _, done = env.execute_action(0)  # simulate a neutral action (we're just observing)
    step += 1

    if done:
        print("\n[Inference] Finished walking through data.")
        break