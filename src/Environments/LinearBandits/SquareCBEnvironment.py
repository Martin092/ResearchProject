from ..AbstractEnvironment import AbstractEnvironment
import numpy as np
import scipy.sparse as sp

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
        if isinstance(self.true_theta, sp.spmatrix):
            reward = self.true_theta.T.dot(feature.reshape((self.d, 1)))[0, 0]
        else:
            reward = np.dot(self.true_theta.flatten(), feature.flatten())
        reward += np.random.normal(loc=0.0, scale=self.sigma)
        return reward

    def generate_context(self):
        return 0

    def generate_theta(self):
        theta = sp.random(self.d, 1, density=self.sparsity)
        norm = sp.linalg.norm(theta)
        if norm > 0:
            theta = theta / norm
        return theta


    def record_regret(self, reward, feature_set):
        best_reward = -float('inf')
        
        for feature in feature_set:
            # Calculate the expected reward (without noise) for this feature
            if isinstance(self.true_theta, sp.spmatrix):
                expected_reward = self.true_theta.T.dot(feature.reshape((self.d, 1)))[0, 0]
            else:
                expected_reward = np.dot(self.true_theta.flatten(), feature.flatten())
                
            best_reward = max(best_reward, expected_reward)

        # Record instantaneous regret
        instantaneous_regret = best_reward - reward
        self.cum_regret += instantaneous_regret
        self.regret.append(instantaneous_regret)

    def observe_actions(self):
        actions = np.random.multivariate_normal(self.means, np.eye(self.k) * self.vars, size=self.d)

        for i in range(self.k):
            if np.linalg.norm(actions[i]) > 0:
                actions[i] = actions[i]
                
        return actions