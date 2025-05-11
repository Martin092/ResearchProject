import numpy as np

from ..AbstractLearner import AbstractLearner
from src.Environments import AbstractEnvironment
from sklearn.linear_model import SGDRegressor
from src.OnlineRegressors.ConcreteRegressors.RidgeRegression import OnlineRidge
from src.OnlineRegressors.ConcreteRegressors.AdaptiveRegression import AdaptiveRegressor
from src.OnlineRegressors.ConcreteRegressors.LinearRegression import LinearRegression
from src.OnlineRegressors.ConcreteRegressors.SGDWrapper import SGDWrapper
from src import OnlineRegressors


class SquareCB(AbstractLearner):
    def __init__(self, T : int, params : dict):
        super().__init__(T, params)
        self.learn_rate = None
        self.mu = None  # exploration parameter
        self.d = None
        # TODO make sure this k is not the same as the sparsity k
        self.k = None

        self.oracle_class = getattr(OnlineRegressors, params["reg"])
        self.oracle = None
        self.params_reg = params["params_reg"]


    def run(self, env : AbstractEnvironment, logger = None):
        self.d = env.get_ambient_dim()
        self.k = env.k
        self.oracle = self.oracle_class(self.d, self.params_reg)
        self.learn_rate = 2 # TODO make not fixed

        for t in range(1, self.T + 1):
            self.action_set = env.observe_actions()
            self.mu = self.k
            context = env.generate_context()
            a_index, action, pred_reward = self.select_action(context)
            features = self.feature_map(action, context)

            reward = env.reveal_reward(features)
            self.oracle.update(features, pred_reward, reward)

            env.record_regret(reward, [self.feature_map(a, context) for a in self.action_set])

            # Log the actions
            if logger is not None:
                logger.log_full(t, context, a_index, action, reward, env.regret[-1])

            self.history.append((action, context, reward))
            self.t += 1

        return env.get_cumulative_regret()


    def feature_map(self, action, context):
        return (action + context).reshape(-1, 1) / np.linalg.norm(action + context)


    def select_action(self, context):
        rewards = np.zeros(len(self.action_set))
        for i, a in enumerate(self.action_set):
            feat = self.feature_map(a, context)
            rewards[i] = self.oracle.predict(feat)

        # TODO make sure to translate the loss to regret correctly
        bt = np.argmax(rewards)
        probabilities = np.zeros(len(self.action_set))
        for i, r in enumerate(rewards):
            if i != bt:
                probabilities[i] = 1 / (self.mu + self.learn_rate * (rewards[bt] - rewards[i]))
        probabilities[bt] = 1 - np.sum(probabilities)
        # print(probabilities)
        index = np.random.choice(np.arange(self.k), size=1, p=probabilities)[0]
        return index, self.action_set[index], rewards[index]


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