from ..AbstractEnvironment import AbstractEnvironment
import numpy as np

class LinearEnvironmentSC(AbstractEnvironment):

    def __init__(self, params):
        super().__init__(params)

        self.theta_high = params["high_theta_vals"]
        if "true_theta" not in params.keys():
            self.true_theta = self.generate_theta()
        else:
            self.true_theta = params["true_theta"]

        self.sigma = params["sigma"]

        # Assuming that the bandit is k-armed
        self.k = params["k"]
        # self.action_set = np.random.rand(self.k, self.d)
        self.action_set = np.eye(self.k, self.d)
        print(self.action_set)


    def reveal_reward(self, action):
        return np.dot(self.true_theta, action.flatten()) + np.random.normal(loc=0.0, scale=self.sigma)

    def generate_context(self):
        # return np.random.rand(self.d)
        return np.ones(self.d)

    def generate_theta(self):
        theta = np.random.rand(self.d)
        indices = np.random.choice(np.arange(self.d), self.theta_high, replace=False)
        for i in range(self.theta_high):
            theta[i] += np.random.rand() * 10
        print(theta)
        return theta

    def record_regret(self, reward, feature_set):
        best_reward = -float('inf')

        for a in feature_set:
            best_reward = max(np.dot(self.true_theta, a), best_reward)

        empirical_regret = best_reward - reward
        self.cum_regret += empirical_regret
        self.regret.append(empirical_regret)

    def observe_actions(self):
        # self.action_set = np.random.normal(0, 1, size=(self.k, self.d))
        # self.action_set = np.ones((self.k, self.d))
        return self.action_set