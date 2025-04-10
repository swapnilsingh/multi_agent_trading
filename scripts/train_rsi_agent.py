import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.trading_env import TradingEnv
from indicators import compute_rsi  # assumes RSI is computed here
from data.market_data import fetch_market_data

# Load data and compute RSI
#df = pd.read_csv("data/BTC.csv")  # Update with your path
df = fetch_market_data(symbol='BTC/USDT', timeframe='1m', limit=10000)
df, _ = compute_rsi(df)
df.dropna(inplace=True)

# Create a custom wrapper to map actions
class MappedTradingEnv(TradingEnv):
    def step(self, action_idx):
        # Map [0, 1, 2] → [-1, 0, 1]
        action = action_idx - 1
        return super().step(action)

# Register vectorized env (needed for PPO)
env = make_vec_env(lambda: MappedTradingEnv(df), n_envs=1)

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Save model
model.save("models/rsi_agent_ppo")
