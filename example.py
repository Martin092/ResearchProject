from environment import Environment, Agent
import numpy as np
import matplotlib
matplotlib.use("tkagg") # delete if that throws error
import matplotlib.pyplot as plt

class BernoulliBandit(Environment):
    def __init__(self, agent, d):
        super().__init__(agent, d)

    def generate_theta(self):
        return np.random.rand(self.d)

    def max_reward(self):
        best_reward = -float('inf')

        for a in self.agent.actions:
            reward = self._reward(self.agent.feature_map(a, self.context()))
            if reward > best_reward:
                best_reward = reward

        return best_reward

    def generate_context(self):
        return np.random.rand(self.agent.K)


class ETC(Agent):
    def __init__(self, actions, T: int):
        super().__init__(actions, T)
        self.optimal_action = self.actions[0]

    def feature_map(self, action, context):
        features = np.zeros(self.env.d)
        features[action] = 1
        return features

    def pick_action(self, context):
        if self.t < self.K:
            return self.actions[self.t]

        if self.t == self.K:
            best_reward = -float('inf')
            for (a, c, r) in self.history:
                if r > best_reward:
                    best_reward = r
                    self.optimal_action = a

        return self.optimal_action


agent = ETC(np.arange(5), 1000)
env = BernoulliBandit(agent, 5)
agent.set_env(env)
res = agent.run()

rewards = []
for (a, c, r) in agent.history:
    rewards.append(r)

plt.title("Rewards over time")
plt.plot(np.cumsum(rewards), label="ETC")
plt.plot(np.cumsum(env.optimal_rewards), label="optimal")
plt.legend()
plt.show()

plt.title("Regret")
plt.plot(np.cumsum(env.optimal_rewards) - np.cumsum(rewards))
plt.show()

