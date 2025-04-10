from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    All agents must implement the methods defined here.
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def act(self, state):
        """Return an action given the current market state."""
        pass

    @abstractmethod
    def train(self, experience):
        """Train the agent using a batch of experience."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Save the model or agent parameters to a file."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Load the model or agent parameters from a file."""
        pass
