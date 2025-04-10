# core/trainers/multi_agent_trainer.py
from core.utils.experience_generator import ExperienceGenerator

class MultiAgentTrainer:
    def __init__(self, agents, env, episodes=10, episode_length=200):
        self.agents = agents
        self.env = env
        self.episodes = episodes
        self.episode_length = episode_length

    def train(self):
        for episode in range(self.episodes):
            print(f"\n[Training] Episode {episode + 1}/{self.episodes}")
            self.env.reset()

            for agent in self.agents:
                generator = ExperienceGenerator(self.env, agent)
                experiences = generator.generate_episode(self.episode_length)

                total_reward = 0
                for exp in experiences:
                    state_id, action, reward, next_state_id = exp
                    agent.train(exp)
                    total_reward += reward

                print(f"[Episode {episode + 1}] {agent.__class__.__name__} total reward: {total_reward:.2f}")
