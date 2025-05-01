from ..AbstractEnvironment import AbstractEnvironment
import numpy as np

class LinearEnvironment(AbstractEnvironment):

    def __init__(self, params):

        super().__init__(params)
        self.action_set = params["action_set"]

        if "true_theta" not in params.keys():
            self.true_theta = self.generate_theta()
        else:
            self.true_theta = params["true_theta"]

        self.sigma = params["sigma"]

        # Assuming that the bandit is k-armed
        self.k = params["k"]

    def reveal_reward(self, action):
        return np.dot(self.true_theta, action) + np.random.normal(loc=0.0, scale=self.sigma)

    def generate_context(self):
        return np.random.rand(self.k)

    def generate_theta(self):
        return np.random.rand(self.d)

    def record_regret(self, reward, feature_set):

        best_reward = -float('inf')

        for a in feature_set:
            best_reward = max(np.dot(self.true_theta, a), best_reward)

        empirical_regret = best_reward - reward
        self.cum_regret += empirical_regret
        self.regret.append(empirical_regret)