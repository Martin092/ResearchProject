from ..AbstractEnvironment import AbstractEnvironment
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class SquareCBEnvironment(AbstractEnvironment):

    def __init__(self, params):
        super().__init__(params)

        self.sparsity = params.get("sparsity", 0.5)
        self.sigma = params.get("sigma", 0.1)

        self.k = params["k"]


        if "true_theta" not in params:
            self.true_theta = self.generate_theta()
        else:
            self.true_theta = params["true_theta"]


        self.best_action = None
        self.best_reward = None

        self.means = np.random.uniform(-2, 2, size=self.k)
        self.vars = np.random.uniform(0, 1, size=self.k) * np.eye(self.k)
    
    def reveal_reward(self, feature):
        assert feature.ndim == 1
        assert self.true_theta.shape[0] == 1
        assert self.true_theta.shape[1] == self.d

        reward = self.true_theta.dot(feature)[0]
        reward += np.random.normal(loc=0.0, scale=self.sigma)
        assert reward.ndim == 0
        return reward

    def generate_context(self):
        return 0

    def generate_theta(self):
        theta = sp.random(1, self.d, density=self.sparsity)
        norm = sp.linalg.norm(theta)
        if norm > 0:
            theta = theta / norm

        assert theta.shape[0] == 1
        assert theta.shape[1] == self.d
        return theta


    def record_regret(self, reward, feature_set):
        assert reward.ndim == 0
        # assert feature_set.shape[1] == self.d

        best_reward = -float('inf')
        
        for feature in feature_set:
            expected_reward = self.true_theta.dot(feature)[0]
            best_reward = max(best_reward, expected_reward)

        # Record instantaneous regret
        instantaneous_regret = best_reward - reward

        assert instantaneous_regret.ndim == 0
        self.cum_regret += instantaneous_regret
        self.regret.append(instantaneous_regret)

    def observe_actions(self):
        actions = np.random.multivariate_normal(self.means, np.eye(self.k) * self.vars, size=self.d).reshape(self.k, self.d)
        return actions

    def feat_mask(self):
        return np.nonzero(self.true_theta.toarray())