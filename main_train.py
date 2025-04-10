# main_train.py
import pandas as pd
from core.environments.multi_agent_trading_environment import MultiAgentTradingEnv
from core.agents.rsi_agent import RSIAgent
from core.agents.macd_agent import MACDAgent
from core.agents.bollinger_agent import BollingerAgent
from core.agents.atr_agent import ATRAgent
from core.agents.sma_agent import SMAAgent
from core.model_management.model_manager import ModelManager
from core.trainers.multi_agent_trainer import MultiAgentTrainer
from core.utils.config_loader import load_config

# Load your dataset (must contain 'Close', 'High', 'Low')
df = pd.read_csv('data/live_binance_ohlcv.csv')

# Initialize ModelManager
model_manager = ModelManager()

# Load configs
rsi_config = load_config("configs/strategies/rsi.yaml")
rsi_config.update({"alpha": 0.1, "gamma": 0.95, "epsilon": 0.2})

macd_config = load_config("configs/strategies/macd.yaml")
macd_config.update({"alpha": 0.1, "gamma": 0.95, "epsilon": 0.2})

bollinger_config = load_config("configs/strategies/bollinger.yaml")
bollinger_config.update({"alpha": 0.1, "gamma": 0.95, "epsilon": 0.2})

atr_config = load_config("configs/strategies/atr.yaml")
atr_config.update({"alpha": 0.1, "gamma": 0.95, "epsilon": 0.2})

sma_config = load_config("configs/strategies/sma.yaml")
sma_config.update({"alpha": 0.1, "gamma": 0.95, "epsilon": 0.2})

# Instantiate agents
agents = [
    RSIAgent(rsi_config, model_manager),
    MACDAgent(macd_config, model_manager),
    BollingerAgent(bollinger_config, model_manager),
    ATRAgent(atr_config, model_manager),
    SMAAgent(sma_config, model_manager)
]

# Create environment
env = MultiAgentTradingEnv(df, window_size=50, initial_balance=1000)

# Train
trainer = MultiAgentTrainer(agents, env, episodes=10, episode_length=200)
trainer.train()

# Save trained models
for agent in agents:
    agent.save_model(f'models/{agent.__class__.__name__}.pkl')
