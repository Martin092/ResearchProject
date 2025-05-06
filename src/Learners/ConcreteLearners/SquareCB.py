import numpy as np

from ..AbstractLearner import AbstractLearner
from src.Environments import AbstractEnvironment
from sklearn.linear_model import SGDRegressor

class SquareCB(AbstractLearner):

    def __init__(self, T : int, params : dict):
        super().__init__(T, params)
        self.learn_rate = params["learn_rate"]
        self.mu = params["mu"] # exploration parameter
        assert self.learn_rate > 0
        assert self.mu > 0
        self.oracle = params["oracle"]()


    def run(self, env : AbstractEnvironment, logger = None):
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
        pass

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