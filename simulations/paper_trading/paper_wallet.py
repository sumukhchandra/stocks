class PaperWallet:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        
    def buy(self, symbol, price, amount_inr):
        if self.balance >= amount_inr:
            qty = amount_inr / price
            self.balance -= amount_inr
            self.positions[symbol] = {
                'entry_price': price,
                'qty': qty,
                'invested': amount_inr
            }
            self.trade_history.append({'action': 'BUY', 'symbol': symbol, 'price': price, 'qty': qty})
            print(f"Bought {qty:.6f} {symbol} at {price}. Balance: {self.balance:.2f}")
            return True
        return False
        
    def sell(self, symbol, price):
        if symbol in self.positions:
            pos = self.positions.pop(symbol)
            revenue = pos['qty'] * price
            profit = revenue - pos['invested']
            self.balance += revenue
            self.trade_history.append({'action': 'SELL', 'symbol': symbol, 'price': price, 'profit': profit})
            print(f"Sold {symbol} at {price}. Profit: {profit:.2f}. Balance: {self.balance:.2f}")
            return True
        return False
