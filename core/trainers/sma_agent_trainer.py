# core/trainers/sma_agent_trainer.py
from core.trainers.base_trainer import BaseTrainer

class SMAAgentTrainer(BaseTrainer):
    def __init__(self, agent, env, config):
        super().__init__(agent, env, config)
        self.name = agent.name

    def train(self, episodes=10):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                state_id = self.agent.get_state_id(self.env.get_current_state())
                action = self.agent.act(state)
                reward, done = self.env.execute_action(action)
                next_state_id = self.agent.get_state_id(self.env.get_current_state())

                #print(f"[{self.name}] Training with state_id: {state_id}, action: {action}, reward: {reward}, next_state_id: {next_state_id}")

                self.agent.learn(state_id, action, reward, next_state_id)
                total_reward += reward

            self.rewards_log.append(total_reward)
            print(f"[Episode {episode}] {self.name} total reward: {total_reward:.2f}")

        self.summarize()
