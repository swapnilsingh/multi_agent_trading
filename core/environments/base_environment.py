from abc import ABC, abstractmethod

class BaseEnvironment(ABC):
    """
    Abstract base class for trading environments.
    """

    @abstractmethod
    def reset(self):
        """
        Reset the environment to its initial state.
        Returns:
            initial_state: the initial state after reset
        """
        pass

    @abstractmethod
    def step(self, action: int):
        """
        Apply the agent's action and advance the environment.

        Args:
            action (int): -1 = sell, 0 = hold, 1 = buy

        Returns:
            next_state: pd.Series or dict representing the next state
            reward: float
            done: bool, whether the episode is over
        """
        pass

    @abstractmethod
    def get_state(self):
        """
        Return the current state of the environment (e.g., market snapshot).
        """
        pass

    @abstractmethod
    def get_reward(self) -> float:
        """
        Return the reward from the most recent action.
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Whether the current episode is complete.
        """
        pass
