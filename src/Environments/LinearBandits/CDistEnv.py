from ..AbstractEnvironment import AbstractEnvironment
import numpy as np
import scipy.sparse as sp


class CDistEnv(AbstractEnvironment):

    def __init__(self, params):
        super().__init__(params)

        self.sparsity = params.get("sparsity", 0.5)
        self.sigma = params.get("sigma", 0.1)

        self.k = params["k"]
        self.action_highest_mean = params["action_highest_mean"]
        self.action_highest_var = params["action_highest_var"]

        self.true_theta = self.generate_theta()

        self.best_action = None
        self.best_reward = None

        self.action_means = np.random.uniform(-self.action_highest_mean,
                                              self.action_highest_mean,
                                              size=self.k)
        self.action_var = np.random.uniform(0,
                                            self.action_highest_var,
                                            size=self.k)
        self.action_set = np.arange(self.k)

    def _generate_fixed_actions(self):
        actions = np.zeros((self.k, self.d))
        for i in range(self.k):
            if i < self.d:
                actions[i, i % self.d] = 1.0
            else:
                idx = np.random.choice(self.d, size=max(1, int(self.d * 0.2)), replace=False)
                actions[i, idx] = np.random.uniform(-1, 1, size=len(idx))
                if np.linalg.norm(actions[i]) > 0:
                    actions[i] = actions[i] / np.linalg.norm(actions[i])
        return actions

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
        """
        Record regret by comparing the received reward to the best possible reward

        Args:
            reward: The reward received for the chosen action
            feature_set: List of feature vectors for all possible actions
        """
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
        actions = np.zeros((self.k, self.d))
        for i in range(self.k):
            actions[i] = np.random.normal(self.action_means[i], self.action_var[i], size=self.d)
        return actions