# scripts/train_bb_agent.py
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from envs.bb_env import BollingerEnv
from indicators import compute_bollinger
from data.market_data import fetch_market_data

# Load market data and compute Bollinger Bands
df = fetch_market_data(symbol='BTC/USDT', timeframe='1m', limit=10000)
df, _ = compute_bollinger(df)
df.dropna(inplace=True)

# Wrap BollingerEnv with vectorized environment
env = make_vec_env(lambda: BollingerEnv(df), n_envs=1)

# Train PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Save model
model.save("models/bb_agent_ppo")
