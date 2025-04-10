# agents/bollinger_agent.py
class BollingerAgent:
    def __init__(self):
        self.name = "Bollinger"

    def preprocess(self, data):
        if len(data) < 2:
            return 0

        prev = data.iloc[-2]
        curr = data.iloc[-1]

        # Detect price crossing above lower band
        if prev['close'] < prev['BBL_20_2.0'] and curr['close'] > curr['BBL_20_2.0']:
            print(f"BollingerAgent: Crossed above lower band → BUY")
            return 1

        # Detect price crossing below upper band
        if prev['close'] > prev['BBU_20_2.0'] and curr['close'] < curr['BBU_20_2.0']:
            print(f"BollingerAgent: Crossed below upper band → SELL")
            return -1

        return 0
