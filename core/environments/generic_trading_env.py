class GenericTradingEnv:
    def __init__(self, initial_capital=1000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []

    def step(self, action, price):
        if action == 1:  # BUY
            if self.capital >= price:
                self.capital -= price
                self.position += 1
                self.trades.append(("BUY", price))
        elif action == -1:  # SELL
            if self.position > 0:
                self.capital += price
                self.position -= 1
                self.trades.append(("SELL", price))
        else:
            self.trades.append(("HOLD", price))
