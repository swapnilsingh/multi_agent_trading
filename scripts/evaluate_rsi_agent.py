import pandas as pd
from stable_baselines3 import PPO
from envs.trading_env import TradingEnv
from indicators import compute_rsi
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv("data/BTC.csv")  # Update path
df, _ = compute_rsi(df)
df.dropna(inplace=True)

class MappedTradingEnv(TradingEnv):
    def step(self, action_idx):
        action = action_idx - 1  # map [0, 1, 2] → [-1, 0, 1]
        return super().step(action)

env = MappedTradingEnv(df)
obs, _ = env.reset()

# Load trained model
model = PPO.load("models/rsi_agent_ppo")

# Evaluate
net_worths = []
actions_log = []
for _ in range(len(df) - env.window_size - 1):
    action_idx, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action_idx)
    env.render()
    net_worths.append(env.net_worth)
    actions_log.append(action_idx - 1)  # Store actual action

    if done:
        break

# Plot Net Worth Over Time
plt.figure(figsize=(12, 6))
plt.plot(net_worths, label="Net Worth")
plt.xlabel("Timestep")
plt.ylabel("Portfolio Value")
plt.title("Evaluation of PPO RSI Agent")
plt.legend()
plt.grid(True)
plt.show()
