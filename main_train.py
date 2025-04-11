import pandas as pd
from core.model_management.model_manager import ModelManager
from core.utils.config_loader import load_config

# Agents
from core.agents.rsi_agent import RSIAgent
from core.agents.macd_agent import MACDAgent
from core.agents.bollinger_agent import BollingerAgent
from core.agents.atr_agent import ATRAgent
from core.agents.sma_agent import SMAAgent

# Trainers
from core.trainers.rsi_agent_trainer import RSIAgentTrainer
from core.trainers.macd_agent_trainer import MACDAgentTrainer
from core.trainers.bollinger_agent_trainer import BollingerAgentTrainer
from core.trainers.atr_agent_trainer import ATRAgentTrainer
from core.trainers.sma_agent_trainer import SMAAgentTrainer

# Environments
from core.environments.rsi_trading_env import RSITRadingEnv
from core.environments.macd_trading_env import MACDTradingEnv
from core.environments.bollinger_trading_env import BollingerTradingEnv
from core.environments.atr_trading_env import ATRTradingEnv
from core.environments.sma_trading_env import SMATradingEnv

# Load your dataset (must contain 'Close', 'High', 'Low')
df = pd.read_csv('data/live_binance_ohlcv.csv')
model_manager = ModelManager()

# ====== Configs ======
rsi_config = load_config("configs/strategies/rsi.yaml")
rsi_config.update({"alpha": 0.1, "gamma": 0.95, "epsilon": 0.2, "model_name": "RSIAgent"})

macd_config = load_config("configs/strategies/macd.yaml")
macd_config.update({"alpha": 0.1, "gamma": 0.95, "epsilon": 0.2, "model_name": "MACDAgent"})

bollinger_config = load_config("configs/strategies/bollinger.yaml")
bollinger_config.update({"alpha": 0.1, "gamma": 0.95, "epsilon": 0.2, "model_name": "BollingerAgent"})

atr_config = load_config("configs/strategies/atr.yaml")
atr_config.update({"alpha": 0.1, "gamma": 0.95, "epsilon": 0.2, "model_name": "ATRAgent"})

sma_config_files = ["configs/strategies/sma_10_30.yaml", "configs/strategies/sma_20_50.yaml"]
sma_configs = []
for path in sma_config_files:
    cfg = load_config(path)
    cfg.update({
        "alpha": 0.1, "gamma": 0.95, "epsilon": 0.2,
        "model_name": f"SMA_{cfg.get('short_window')}_{cfg.get('long_window')}"
    })
    sma_configs.append(cfg)

# ====== Agents and Trainers ======
trainers = [
    RSIAgentTrainer(RSIAgent(rsi_config, model_manager),
                    RSITRadingEnv(df, window_size=50, initial_balance=1000, rsi_period=rsi_config.get("rsi_period", 14)),
                    rsi_config),

    MACDAgentTrainer(MACDAgent(macd_config, model_manager),
                     MACDTradingEnv(df, window_size=50, initial_balance=1000),
                     macd_config),

    BollingerAgentTrainer(BollingerAgent(bollinger_config, model_manager),
                          BollingerTradingEnv(df, window_size=50, initial_balance=1000),
                          bollinger_config),

    ATRAgentTrainer(ATRAgent(atr_config, model_manager),
                    ATRTradingEnv(df, window_size=50, initial_balance=1000),
                    atr_config),
]

# Add SMA trainers
for config in sma_configs:
    sma_agent = SMAAgent(config, model_manager)
    sma_env = SMATradingEnv(df, window_size=50, initial_balance=1000)
    trainers.append(SMAAgentTrainer(sma_agent, sma_env, config))

# ====== Training ======
for trainer in trainers:
    trainer.train(episodes=10)
    trainer.agent.save_model(f"models/{trainer.agent.name}.pkl")
