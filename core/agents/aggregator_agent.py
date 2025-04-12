# core/agents/aggregator_agent.py
from core.agents.base_agent import BaseAgent

class AggregatorAgent(BaseAgent):
    def __init__(self, agents, model_manager):
        super().__init__({}, model_manager)  # no config needed for aggregator
        self.agents = agents
        self.required_window = 0  # Aggregator doesn’t need data window

    def act(self, state):
        agent_actions = []

        for agent in self.agents:
            action = agent.act(state)
            if action is not None:
                agent_actions.append(action)
                print(f"[Aggregator] {agent.__class__.__name__} → action: {action}")
            else:
                print(f"[Aggregator] {agent.__class__.__name__} → skipped (not enough data)")

        return self.make_final_decision(agent_actions)

    def make_final_decision(self, agent_actions):
        if not agent_actions:
            print("[Aggregator] No valid agent actions. Defaulting to HOLD (0).")
            return 0  # HOLD

        buy_votes = agent_actions.count(1)
        sell_votes = agent_actions.count(-1)

        if buy_votes > sell_votes:
            return 1
        elif sell_votes > buy_votes:
            return -1
        else:
            return 0  # HOLD if tie

    def save_model(self, filepath):
        pass

    def load_model(self, filepath):
        pass

    def train(self, experience):
        pass

    def learn(self, *args, **kwargs):  # ✅ added to fulfill abstract method requirement
        pass

    def vote_from_actions(self, action_dict):
        """
        Accepts a dictionary of actions from individual agents and returns the majority vote.
        """
        print("[Aggregator] Received agent actions:")
        for name, action in action_dict.items():
            print(f"    → {name.upper()}: {action}")
        return self.make_final_decision(list(action_dict.values()))

