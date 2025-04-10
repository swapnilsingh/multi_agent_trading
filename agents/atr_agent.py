# atr_agent.py
from agents.base_agent import BaseRLAgent
from indicators import compute_atr

class ATRAgent(BaseRLAgent):
    def preprocess(self, data):
        data, atr = compute_atr(data)
        return [atr]

    def choose_action(self, state):
        atr = state[0]
        if atr > 1.5:
            return 'sell'
        elif atr < 0.5:
            return 'buy'
        return 'hold'
