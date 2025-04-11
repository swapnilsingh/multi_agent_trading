# core/trainers/base_trainer.py
import numpy as np

class BaseTrainer:
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config
        self.rewards_log = []  # Store rewards for summary
        self.agent.state_action_counter = {}  # Reset action frequencies for each training run

    def train(self, episodes=10):
        """
        The train method is meant to be implemented by the subclass.
        """
        raise NotImplementedError("Each trainer must implement the train method")

    def summarize(self):
        """
        Summarize the training results.
        """
        total_rewards = sum(self.rewards_log)
        mean_reward = np.mean(self.rewards_log)
        max_reward = np.max(self.rewards_log)
        min_reward = np.min(self.rewards_log)
        last_10_rewards = self.rewards_log[-10:]

        print(f"=== {self.agent.name} Training Summary ===")
        print(f"- Total Rewards Logged: {total_rewards}")
        print(f"- Mean Reward: {mean_reward:.2f}")
        print(f"- Max Reward: {max_reward:.2f}")
        print(f"- Min Reward: {min_reward:.2f}")
        print(f"- Last 10 Rewards: {last_10_rewards}")

        # Action frequencies by state
        print(f"\n{self.agent.name} Action Frequencies by State:")
        for state_id, action in self.agent.state_action_counter.keys():
            count = self.agent.state_action_counter[(state_id, action)]
            print(f" - State: {state_id}, Action: {action}, Count: {count}")

    def log_reward(self, reward):
        """
        Log the reward during training.
        """
        self.rewards_log.append(reward)
