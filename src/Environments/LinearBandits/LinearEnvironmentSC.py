from ..AbstractEnvironment import AbstractEnvironment
import numpy as np

class LinearEnvironmentSC(AbstractEnvironment):

    def __init__(self, params):

        super().__init__(params)
        self.actions = params["actions"]

        if "theta_sparsity" not in params.keys():
            self.theta_sparsity = int(self.d * 0.3)
        else:
            self.theta_sparsity = params["theta_sparsity"]

        if "true_theta" not in params.keys():
            self.true_theta = self.generate_theta()
        else:
            self.true_theta = params["true_theta"]

        self.sigma = params["sigma"]

        # Assuming that the bandit is k-armed
        self.k = params["k"]

    def reveal_reward(self, action):
        return np.dot(self.true_theta, action.flatten()) + np.random.normal(loc=0.0, scale=self.sigma)

    def generate_context(self):
        return np.random.rand(self.k)

    def generate_theta(self):
        theta = np.zeros(self.d)
        indices = np.random.choice(np.arange(self.d), self.theta_sparsity, replace=False)
        for i in indices:
            theta[i] = np.random.rand()
        return theta

    def record_regret(self, reward, feature_set):
        best_reward = -float('inf')

        for a in feature_set:
            best_reward = max(np.dot(self.true_theta, a), best_reward)

        empirical_regret = best_reward - reward
        self.cum_regret += empirical_regret
        self.regret.append(empirical_regret)

    def observe_actions(self):
        self.action_set = np.random.normal(0, 5, size=(self.actions, self.d))
        self.action_set[0] = np.random.normal(2, 5, size=(1, self.d))
        return self.action_set