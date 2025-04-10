# core/envs/multi_agent_trading_env.py
import pandas as pd

class MultiAgentTradingEnv:
    def __init__(self, df: pd.DataFrame, window_size=50, initial_balance=1000):
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # +1 for long, -1 for short, 0 for neutral
        self.entry_price = None
        return self.df.iloc[:self.current_step]

    def get_current_state(self):
        window = self.df.iloc[self.current_step - self.window_size : self.current_step]
        return {
            'close_history': window['Close'].tolist(),
            'high_history': window['High'].tolist(),
            'low_history': window['Low'].tolist(),
        }

    def execute_action(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0

        if action == 1:  # BUY
            if self.position == 0:
                self.entry_price = current_price
                self.position = 1
            elif self.position == -1:
                reward = self.entry_price - current_price
                self.balance += reward
                self.position = 0
                self.entry_price = None

        elif action == -1:  # SELL
            if self.position == 0:
                self.entry_price = current_price
                self.position = -1
            elif self.position == 1:
                reward = current_price - self.entry_price
                self.balance += reward
                self.position = 0
                self.entry_price = None

        self.current_step += 1
        done = self.current_step >= len(self.df)
        return reward, done

    def skip(self):
        self.current_step += 1
