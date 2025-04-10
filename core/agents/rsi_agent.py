# agents/rsi_agent.py
class RSIAgent:
    def __init__(self, name="RSI", lower=30, upper=70):
        self.name = name
        self.lower = lower
        self.upper = upper

    def preprocess(self, data):
        if len(data) < 2:
            return 0

        prev_rsi = data.iloc[-2]['rsi']
        curr_rsi = data.iloc[-1]['rsi']

        # Crossover above lower bound → BUY signal
        if prev_rsi < self.lower and curr_rsi >= self.lower:
            print(f"RSIAgent: RSI crossed above {self.lower} → BUY")
            return 1

        # Crossover below upper bound → SELL signal
        elif prev_rsi > self.upper and curr_rsi <= self.upper:
            print(f"RSIAgent: RSI crossed below {self.upper} → SELL")
            return -1

        # RSI in neutral range → HOLD
        return 0
