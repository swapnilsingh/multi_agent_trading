# core/agents/base_agent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.name = config.get("model_name", self.__class__.__name__)
        self.required_window = None
        self.state_action_counter = {}

    def has_sufficient_data(self, data):
        if self.required_window is None:
            raise NotImplementedError("Subclasses must define required_window.")
        return len(data) >= self.required_window

    @abstractmethod
    def act(self, state):
        pass

    @abstractmethod
    def learn(self, state_id, action, reward, next_state_id):
        pass

    @abstractmethod
    def save_model(self, filepath):
        pass

    @abstractmethod
    def load_model(self, filepath):
        pass
