# core/trainers/rsi_agent_trainer.py
from core.trainers.base_trainer import BaseTrainer

class RSIAgentTrainer(BaseTrainer):
    def __init__(self, agent, env, config):
        super().__init__(agent, env, config)
        self.rsi_period = config.get("rsi_period", 14)

    def _get_state_id(self, state):
        close_history = state.get("close_history", [])
        if isinstance(close_history[0], list):
            close_history = [x[0] for x in close_history]

        import pandas as pd
        close_series = pd.Series(close_history).astype(float)
        delta = close_series.diff().fillna(0)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=self.rsi_period).mean().iloc[-1]
        avg_loss = loss.rolling(window=self.rsi_period).mean().iloc[-1]

        if avg_loss == 0:
            return "rsi_100"

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return f"rsi_{int(rsi) // 10}"

    def train(self, episodes=10):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                state_id = self._get_state_id(state)
                action = self.agent.act(state_id)
                reward, done = self.env.execute_action(action)
                next_state = self.env.get_current_state()
                next_state_id = self._get_state_id(next_state)

                self.agent.learn(state_id, action, reward, next_state_id)
                total_reward += reward

            self.rewards_log.append(total_reward)
            print(f"[Episode {episode}] {self.agent.name} total reward: {total_reward:.2f}")

        self.summarize()
