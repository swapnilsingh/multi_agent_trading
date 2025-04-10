# base_agent.py
class BaseRLAgent:
    def __init__(self, name):
        self.name = name
        self.model = None  # Replace with actual RL model

    def preprocess(self, data):
        raise NotImplementedError

    def choose_action(self, state):
        raise NotImplementedError

    def train(self, env):
        raise NotImplementedError
