# agents/sma_agent.py
class SMA_Agent:
    def __init__(self, df, short_window=5, long_window=20):
        self.df = df
        self.short_window = short_window
        self.long_window = long_window

    def act(self, step):
        if step < self.long_window:
            return 0
        short_sma = self.df['close'].iloc[step - self.short_window:step].mean()
        long_sma = self.df['close'].iloc[step - self.long_window:step].mean()

        if short_sma > long_sma:
            print(f"SMA Agent: Short SMA > Long SMA ({short_sma:.2f} > {long_sma:.2f}) → BUY")
            return 1
        elif short_sma < long_sma:
            print(f"SMA Agent: Short SMA < Long SMA ({short_sma:.2f} < {long_sma:.2f}) → SELL")
            return -1
        return 0
