from ..AbstractEnvironment import AbstractEnvironment
import numpy as np
import scipy.sparse as sp

class SparseLinearEnvironment(AbstractEnvironment):

    def __init__(self, params):

        super().__init__(params)
        self.actions = params["actions"]
        self.sparsity = params["sparsity"]

        if "true_theta" not in params.keys():
            self.true_theta = self.generate_theta()
        else:
            self.true_theta = params["true_theta"]

        self.sigma = params["sigma"]

        # Assuming that the bandit is k-armed
        self.k = params["k"]

    def reveal_reward(self, action):
        # sparse arrays being cringe or me being stupid
        return self.true_theta.T.dot(action.reshape((self.d, 1)))[0][0] + np.random.normal(loc=0.0, scale=self.sigma)

    def generate_context(self):
        return np.random.rand(self.k)

    def generate_theta(self):
        return sp.random(self.d, 1, self.sparsity)

    def record_regret(self, reward, feature_set):
        best_reward = -float('inf')
        for a in feature_set:
            best_reward = max(self.true_theta.T.dot(a.reshape((self.d, 1)))[0][0], best_reward)

        empirical_regret = best_reward - reward
        self.cum_regret += empirical_regret
        self.regret.append(empirical_regret)

    def observe_actions(self):
        self.action_set = np.random.normal(0, 1, size=(self.actions, self.d))
        return self.action_set