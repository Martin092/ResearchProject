import numpy as np
from sklearn.linear_model import SGDRegressor

from src.Environments import AbstractEnvironment
from src.Learners import AbstractLearner
from src.OnlineRegressors.ConcreteRegressors.OnlineRidgeFSSCB import RidgeFSSCB


class ThomsonSampling(AbstractLearner):

    def __init__(self, T : int, params : dict):
        super().__init__(T, params)

        self.optimal_action = None
        self.action_set = None
        self.d = None
        self.k = None
        self.regressor = None
        self.regularization = params["regularization"]

        self.V = None


    def run(self, env: AbstractEnvironment, logger=None):
        self.d = env.get_ambient_dim()
        self.k = env.k

        self.V = self.regularization * np.eye(self.d)
        # regressor is used to estimate theta, which we can then use in the prior as the mean of the distribution
        # the regressor has w = 0 in the beginning, which is our prior - p(theta) = N(0, lambda * I)
        self.regressor = RidgeFSSCB(self.d, {"lambda_reg": self.regularization})


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
        theta = np.random.multivariate_normal(self.regressor.w.flatten(), np.linalg.inv(self.V)).reshape(-1, 1)

        reward = -float('inf')
        action_index = 0
        for i, a in enumerate(self.action_set):
            new_reward = a.T @ theta
            if new_reward > reward:
                reward = new_reward
                action_index = i

        return self.action_set[action_index]

    def update(self, feat, reward):
        self.V += feat @ feat.T
        self.regressor.predict_and_fit(feat, reward)

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