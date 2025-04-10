# scripts/training_manager.py
class TrainingManager:
    def __init__(self, agents, aggregator_agent, model_manager):
        self.agents = agents
        self.aggregator_agent = aggregator_agent
        self.model_manager = model_manager
        self.training_steps = 0

    def train_agents(self, state):
        # Train each agent on the current state
        for agent in self.agents:
            agent.train(state)
        
        # After training, aggregate the results and tune the model
        self.aggregator_agent.train(state)
        self.model_manager.save_models(self.agents)

    def run(self, environment):
        while self.training_steps < 1000:
            state = environment.reset()
            done = False
            while not done:
                agent_actions = [agent.act(state) for agent in self.agents]
                final_action = self.aggregator_agent.make_final_decision(agent_actions)
                state, reward, done = environment.step(agent_actions)
                self.train_agents(state)
                self.training_steps += 1
