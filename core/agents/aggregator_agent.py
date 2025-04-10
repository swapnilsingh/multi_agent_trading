# core/agents/aggregator_agent.py
from core.agents.base_agent import BaseAgent

class AggregatorAgent(BaseAgent):
    def __init__(self, agents):
        super().__init__(None)
        self.agents = agents

    def act(self, state):
        agent_actions = [agent.act(state) for agent in self.agents]
        return self.make_final_decision(agent_actions)

    def make_final_decision(self, agent_actions):
        # Example decision-making logic (you can modify this)
        if agent_actions.count(1) > agent_actions.count(-1):
            return 1  # Buy
        else:
            return -1  # Sell

    def save_model(self, filepath):
        pass  # No model to save for the aggregator

    def load_model(self, filepath):
        pass  # No model to load for the aggregator

    def train(self, experience):
        pass  # Aggregator doesn't require training
