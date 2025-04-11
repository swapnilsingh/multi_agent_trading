# core/trainers/atr_agent_trainer.py
from core.trainers.base_trainer import BaseTrainer

class ATRAgentTrainer(BaseTrainer):
    def __init__(self, agent, env, config):
        super().__init__(agent, env, config)
        self.name = agent.name

    def train(self, episodes=10):
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.agent.act(state)
                reward, done = self.env.execute_action(action)
                next_state = self.env.get_current_state()

                try:
                    high = state.get("high_history", [])
                    low = state.get("low_history", [])
                    close = state.get("close_history", [])
                    next_high = next_state.get("high_history", [])
                    next_low = next_state.get("low_history", [])
                    next_close = next_state.get("close_history", [])

                    atr = self.agent.calculate_atr(high, low, close)
                    next_atr = self.agent.calculate_atr(next_high, next_low, next_close)

                    state_id = self.agent.get_state_id(atr)
                    next_state_id = self.agent.get_state_id(next_atr)

                    self.agent.learn(state_id, action, reward, next_state_id)
                except Exception as e:
                    print(f"[{self.name}] ⚠️ Error during training step: {e}")
                    continue

                total_reward += reward
                state = next_state

            rewards.append(total_reward)
            print(f"[Episode {episode}] {self.name} total reward: {total_reward:.2f}")

        self.rewards_log = rewards
        self.summarize()
        return rewards
