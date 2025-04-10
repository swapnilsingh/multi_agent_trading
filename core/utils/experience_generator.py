# core/utils/experience_generator.py
import random

class ExperienceGenerator:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def generate_episode(self, episode_length=100):
        """
        Simulates a single trading episode and collects experience tuples:
        (state_id, action, reward, next_state_id)
        """
        history = self.env.reset()
        episode_experience = []

        for t in range(episode_length):
            state = self.env.get_current_state()
            action = self.agent.act(state)

            if action is None:
                self.env.skip()
                continue

            reward, done = self.env.execute_action(action)
            next_state = self.env.get_current_state()

            state_id = self.agent.get_state_id(state['close_history'])
            next_state_id = self.agent.get_state_id(next_state['close_history'])

            episode_experience.append((state_id, action, reward, next_state_id))

            if done:
                break

        return episode_experience
