from ..AbstractEnvironment import AbstractEnvironment
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg


class DistEnvFixedActions(AbstractEnvironment):
    def __init__(self, params):
        super().__init__(params)

        self.sparsity = params["sparsity"]
        self.noise = params["noise"]

        # Assuming that the bandit is k-armed
        self.k = params["k"]

        # each action is a 2d array that holds the mean and variance of the normal distribution that will be sampled
        self.action_set = (np.random.rand(self.k, 2))
        self.action_set[:, 0] -= 0.5
        self.action_set[:, 0] *= 1
        self.action_set[:, 1] *= 2
        # self.action_set[0][0] = 3
        # self.action_set[0][1] = 4
        self.true_theta = self.generate_theta()

    def reveal_reward(self, feature):
        assert self.true_theta.shape == (self.d, 1)
        assert feature.shape == (self.d, 1)
        return (self.true_theta.T @ feature + np.random.normal(loc=0.0, scale=self.noise))[0][0]

    def generate_context(self):
        return np.random.rand(self.d)

    def generate_theta(self):
        theta = sp.random(self.d, 1, density=self.sparsity)
        print(theta)
        return theta / np.linalg.norm(theta.toarray())

    def record_regret(self, reward, feature_set):
        assert self.true_theta.shape == (self.d, 1)
        assert np.array(feature_set).shape == (self.k, self.d, 1)
        best_reward = -float('inf')

        for a in feature_set:
            best_reward = max(self.true_theta.T @ a, best_reward)

        empirical_regret = best_reward[0][0] - reward
        self.cum_regret += empirical_regret
        self.regret.append(empirical_regret)

    def observe_actions(self):
        actions = np.zeros((self.k, self.d))
        for i in range(self.k):
            actions[i] = np.random.normal(self.action_set[i][0], self.action_set[i][1], size=(self.d))
        return actions
