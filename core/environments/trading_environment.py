import pandas as pd
from core.environments.base_environment import BaseEnvironment

class TradingEnvironment(BaseEnvironment):
    def __init__(self, market_data: pd.DataFrame, initial_balance: float = 10000):
        """
        Args:
            market_data (pd.DataFrame): OHLCV + indicator data
            initial_balance (float): Starting capital for the agent
        """
        self.market_data = market_data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # 1 = long, -1 = short, 0 = flat
        self.entry_price = 0
        self.trades = []
        self.done = False
        return self.get_state()

    def step(self, action: int):
        """
        Takes an action: -1 = sell, 0 = hold, 1 = buy
        Applies it and moves forward one step
        """
        if self.done:
            return self.get_state(), 0.0, self.done

        price = self.market_data.loc[self.current_step, "close"]

        reward = 0.0

        # Execute action
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == -1 and self.position == 0:
            self.position = -1
            self.entry_price = price
        elif action == 0:
            pass  # Hold
        elif action == -1 and self.position == 1:
            reward = price - self.entry_price
            self.balance += reward
            self.trades.append(("SELL", price, reward))
            self.position = 0
        elif action == 1 and self.position == -1:
            reward = self.entry_price - price
            self.balance += reward
            self.trades.append(("COVER", price, reward))
            self.position = 0

        self.current_step += 1
        if self.current_step >= len(self.market_data):
            self.done = True

        return self.get_state(), reward, self.done

    def get_state(self):
        """
        Returns the current market snapshot for this step.
        Could return just the row, or a custom subset.
        """
        return self.market_data.iloc[self.current_step] if not self.done else pd.Series()

    def get_reward(self) -> float:
        # Could return PnL delta, Sharpe-based reward, etc.
        return self.balance - self.initial_balance

    def is_terminal(self) -> bool:
        return self.done
