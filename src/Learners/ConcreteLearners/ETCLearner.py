import numpy as np

from ..AbstractLearner import AbstractLearner
from src.Environments import LinearEnvironment
from sklearn.linear_model import SGDRegressor

class ETCLearner(AbstractLearner):

    def __init__(self, T : int, params : dict):
        super().__init__(T, params)

        self.optimal_action = None
        self.action_set = None
        self.d = None
        self.k = None
        self.N = params["N"]
        self.regressor = None


    def run(self, env : LinearEnvironment, logger = None):
        self.d = env.get_ambient_dim()
        self.k = env.k
        self.regressor = SGDRegressor(penalty='l2', alpha=0.01)

        for t in range(1, self.T + 1):
            self.action_set = env.observe_actions()
            context = env.generate_context()
            action = self.select_action(context)
            features = self.feature_map(action, context).reshape(1, -1)

            reward = env.reveal_reward(features)
            self.regressor.partial_fit(features, [reward])

            env.record_regret(reward, [self.feature_map(a, context) for a in self.action_set])

            # Log the actions
            if logger is not None:
                logger.log(t, reward, env.regret[-1])

            self.history.append((action, context, reward))
            self.t += 1

        return env.get_cumulative_regret()


    def feature_map(self, action, context):
        return np.array(action)


    def select_action(self, context):
        if self.t < self.N * self.k:
            return self.action_set[self.t % self.k]

        if self.t == self.N * self.k:
            best_reward = -float('inf')
            for a in self.action_set:
                r = self.regressor.predict(self.feature_map(a, context).reshape(1, -1))
                if r > best_reward:
                    best_reward = r
                    self.optimal_action = a

        return self.optimal_action

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