import numpy as np
from sklearn.linear_model import SGDRegressor

from src.Environments import AbstractEnvironment
from src.Learners import AbstractLearner


class LinUCBLearner(AbstractLearner):

    def __init__(self, T : int, params : dict):
        super().__init__(T, params)

        self.optimal_action = None
        self.action_set = None
        self.d = None
        self.k = None
        self.regressor = None
        self.delta = params["delta"]
        self.regularization = params["regularization"]

        self.b = None
        self.V = None
        self.theta = None


    def run(self, env: AbstractEnvironment, logger=None):
        self.d = env.get_ambient_dim()
        self.k = env.k

        self.b = np.zeros(self.d).reshape(-1, 1)
        self.V = self.regularization * np.eye(self.d)
        self.theta = np.zeros((self.d, 1))

        for t in range(1, self.T + 1):
            self.action_set = env.observe_actions()
            context = env.generate_context()
            action = self.select_action(context)
            features = self.feature_map(action, context)

            reward = env.reveal_reward(features)
            self.update(features, reward)

            env.record_regret(reward, [self.feature_map(a, context) for a in self.action_set])

            # Log the actions
            if logger is not None:
                logger.log(t, reward, env.regret[-1])

            self.history.append((action, context, reward))
            self.t += 1

        return env.get_cumulative_regret()

    def select_action(self, context):
        beta = np.sqrt(self.regularization)
        beta += np.sqrt(2 * np.log(1/self.delta) + self.d * (np.log(1 + (self.t)/(self.regularization * self.d))))

        V_inv = np.linalg.inv(self.V)
        max_ucb = -float('inf')
        best_action = self.action_set[0]
        for a in self.action_set:
            feat = self.feature_map(a, context)
            ucb = feat.T @ self.theta + beta * np.sqrt(feat.T @ V_inv @ feat)
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = a

        return best_action

    def update(self, feat, reward):
        self.V += feat @ feat.T
        self.b += reward * feat
        self.theta = np.linalg.inv(self.V) @ self.b

    def total_reward(self):
        total = 0
        for (a, c, r) in self.history:
            total += r
        return total

    def cum_reward(self):
        total = []
        for (a, c, r) in self.history:
            total.append(r)
        return total

    def feature_map(self, action, context):
        return (action).reshape(-1, 1)