from abc import ABC, abstractmethod
import numpy as np

class Environment(ABC):
    def __init__(self, agent, d):
        self.agent = agent
        self.d = d
        self.contexts = [self.generate_context()]
        self.optimal_rewards = []
        self._theta = self.generate_theta()

    @abstractmethod
    def generate_theta(self):
        pass

    @abstractmethod
    def generate_context(self):
        pass

    @abstractmethod
    def max_reward(self):
        pass

    def _reward(self, feature):
        return np.dot(feature, self._theta) + np.random.normal()

    def context(self):
        return self.contexts[-1]

    def pull(self, feature):
        reward = self._reward(feature)
        self.optimal_rewards.append(self.max_reward())
        self.contexts.append(self.generate_context())
        return reward


class Agent(ABC):
    def __init__(self, actions, T: int):
        self.T = T
        self.t = 0
        self.actions = actions
        self.K = len(actions)
        self.history = []

    def set_env(self, env: Environment):
        self.env = env

    @abstractmethod
    def feature_map(self, action, context):
        pass

    @abstractmethod
    def pick_action(self, context):
        pass

    def total_reward(self):
        total = 0
        for (a, c, r) in self.history:
            total += r
        return total

    def pull(self, action):
        context = self.env.context()
        features = self.feature_map(action, context)
        reward = self.env.pull(features)
        self.history.append((action, context, reward))
        return reward

    def run(self):
        for i in range(self.T):
            self.pull(
                self.pick_action(self.env.context())
            )
            self.t += 1
        return np.sum(self.env.optimal_rewards) - self.total_reward()
