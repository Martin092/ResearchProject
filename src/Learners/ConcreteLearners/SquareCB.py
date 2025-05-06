import numpy as np

from ..AbstractLearner import AbstractLearner
from src.Environments import AbstractEnvironment
from sklearn.linear_model import SGDRegressor
from src.OnlineRegressors.ConcreteRegressors.RidgeRegression import OnlineRidge
from src.OnlineRegressors.ConcreteRegressors.AdaptiveRegression import AdaptiveRegressor
from src.OnlineRegressors.ConcreteRegressors.LinearRegression import LinearRegression

class SquareCB(AbstractLearner):

    def __init__(self, T : int, params : dict):
        super().__init__(T, params)
        self.learn_rate = None
        self.mu = None # exploration parameter

        self.oracle = None


    def run(self, env : AbstractEnvironment, logger = None):
        self.d = env.get_ambient_dim()
        self.k = env.k
        self.oracle = LinearRegression(self.d, 0.01)
        self.learn_rate = 0.01

        for t in range(1, self.T + 1):
            self.action_set = env.observe_actions()
            self.mu = len(self.action_set)
            context = env.generate_context()
            a_index, action, pred_reward = self.select_action(context)
            features = self.feature_map(action, context).reshape(1, -1)

            reward = env.reveal_reward(features)
            self.oracle.update(features.reshape(-1, 1), pred_reward, reward)

            env.record_regret(reward, [self.feature_map(a, context) for a in self.action_set])

            # Log the actions
            if logger is not None:
                logger.log_full(t, context, a_index, action, reward, env.regret[-1])

            self.history.append((action, context, reward))
            self.t += 1

        return env.get_cumulative_regret()


    def feature_map(self, action, context):
        return np.array(action) + context


    def select_action(self, context):
        rewards = np.zeros(len(self.action_set))
        for i, a in enumerate(self.action_set):
            feat = self.feature_map(a, context)
            rewards[i] = self.oracle.predict(feat)

        bt = np.argmin(rewards)
        probabilities = np.zeros(len(self.action_set))
        for i, r in enumerate(rewards):
            if i != bt:
                probabilities[i] = 1 / (self.mu + self.learn_rate * (rewards[i] - rewards[bt]))
        probabilities[bt] = 1 - np.sum(probabilities)
        index = np.random.choice(np.arange(len(self.action_set)), size=1, p=probabilities)
        return index[0], self.action_set[index[0]], rewards[index]


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