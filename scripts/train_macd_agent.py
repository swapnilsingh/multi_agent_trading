import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.macd_env import MACDTradingEnv
from indicators import compute_macd
from data.market_data import fetch_market_data

# Step 1: Fetch + prepare market data
df = fetch_market_data(symbol='BTC/USDT', timeframe='1m', limit=10000)
df, macd_data = compute_macd(df)
df.dropna(inplace=True)

# Step 2: Wrap in vectorized environment for PPO
class MappedMACDEnv(MACDTradingEnv):
    def step(self, action_idx):
        action = action_idx - 1
        return super().step(action)

env = make_vec_env(lambda: MappedMACDEnv(df), n_envs=1)

# Step 3: Train PPO agent
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/macd/")
model.learn(total_timesteps=10_000)

# Step 4: Save model
model.save("models/macd_agent_ppo")
print("✅ MACD Agent trained and saved as macd_agent_ppo.zip")
