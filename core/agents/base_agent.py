# core/agents/base_agent.py

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    All agents must implement the methods defined here.
    """

    def __init__(self, config):
        self.config = config
        self.required_window = None  # <-- add this line

    def has_sufficient_data(self, data):
        """
        Checks if enough data is available to compute indicators.
        Agents should call this before acting.
        """
        if self.required_window is None:
            raise NotImplementedError("Subclasses must define required_window.")
        return len(data) >= self.required_window

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def save_model(self, filepath):
        pass

    @abstractmethod
    def load_model(self, filepath):
        pass

    @abstractmethod
    def train(self, experience):
        pass
