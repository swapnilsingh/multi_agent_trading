# agents/macd_agent.py
class MACDAgent:
    def __init__(self, name="MACD"):
        self.name = name

    def preprocess(self, data):
        if 'MACD_12_26_9' not in data.columns or 'MACDs_12_26_9' not in data.columns:
            return 0

        prev_macd = data.iloc[-2]['MACD_12_26_9']
        prev_signal = data.iloc[-2]['MACDs_12_26_9']
        curr_macd = data.iloc[-1]['MACD_12_26_9']
        curr_signal = data.iloc[-1]['MACDs_12_26_9']

        # MACD line crosses above signal line → BUY
        if prev_macd < prev_signal and curr_macd >= curr_signal:
            print("MACDAgent: MACD crossover ↑ → BUY")
            return 1
        # MACD line crosses below signal line → SELL
        elif prev_macd > prev_signal and curr_macd <= curr_signal:
            print("MACDAgent: MACD crossover ↓ → SELL")
            return -1

        return 0
