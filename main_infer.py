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

df = pd.read_csv("data/live_binance_ohlcv.csv")
env = MultiAgentTradingEnv(df, window_size=50, initial_balance=1000)
model_manager = ModelManager()

rsi_config = load_config("configs/strategies/rsi.yaml")
rsi_config.update({"model_name": "RSIAgent"})

macd_config = load_config("configs/strategies/macd.yaml")
macd_config.update({"model_name": "MACDAgent"})

bollinger_config = load_config("configs/strategies/bollinger.yaml")
bollinger_config.update({"model_name": "BollingerAgent"})

atr_config = load_config("configs/strategies/atr.yaml")
atr_config.update({"model_name": "ATRAgent"})

sma_config_files = ["configs/strategies/sma_10_30.yaml", "configs/strategies/sma_20_50.yaml"]
sma_configs = []
for path in sma_config_files:
    cfg = load_config(path)
    cfg.update({"model_name": f"SMA_{cfg.get('short_window')}_{cfg.get('long_window')}"})
    sma_configs.append(cfg)

agents = [
    RSIAgent(rsi_config, model_manager),
    MACDAgent(macd_config, model_manager),
    BollingerAgent(bollinger_config, model_manager),
    ATRAgent(atr_config, model_manager),
]

for config in sma_configs:
    agents.append(SMAAgent(config, model_manager))

for agent in agents:
    model_path = f"models/{agent.name}.pkl"
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"[Model] Loaded: {model_path}")
    else:
        print(f"[Warning] No model found for {agent.name}, using default params.")

env.reset()
step = 0

while True:
    state = env.get_current_state()
    print(f"\n[Step {step}] Close Price: {state['close_history'][-1]}")

    for agent in agents:
        action = agent.act(state)
        action_str = "BUY" if action == 1 else "SELL" if action == -1 else "HOLD" if action == 0 else "WAIT"
        print(f"{agent.name}: {action_str}")

    _, done = env.execute_action(0)
    step += 1

    if done:
        print("\n[Inference] Finished walking through data.")
        break
