# core/oms.py
class OrderManagementSystem:
    def __init__(self, initial_capital):
        self.capital = initial_capital
        self.inventory = 0.0  # use float for fractional inventory
        self.trade_history = []
        self.equity_curve = []

    def execute(self, action, price):
        if action == 1:
            units = self.capital / price  # buy as much as you can
            self.inventory += units
            self.capital -= units * price
            self.trade_history.append(('BUY', price, units))
        elif action == -1 and self.inventory > 0:
            self.capital += self.inventory * price
            self.trade_history.append(('SELL', price, self.inventory))
            self.inventory = 0.0

    def update_equity(self, price):
        equity = self.capital + self.inventory * price
        self.equity_curve.append(equity)

    def summary(self):
        return {
            'Final Capital': round(self.capital + self.inventory * df.iloc[-1]['close'], 2),
            'Inventory Left': round(self.inventory, 6),
            'Total Trades': len(self.trade_history)
        }
