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
        self.learn_rate = params["learn_rate"]
        self.mu = None
        self.d = None
        # TODO make sure this k is not the same as the sparsity k
        self.k = None

        self.oracle_class = getattr(OnlineRegressors, params["reg"])
        self.oracle = None
        self.params_reg = params["params_reg"]

        self.alpha = params.get("alpha", 1.0)


    def run(self, env : AbstractEnvironment, logger = None):
        self.d = env.get_ambient_dim()
        self.k = env.k
        self.oracle = self.oracle_class(self.d, self.params_reg)
        self.mu = self.k 

        for t in range(1, self.T + 1):
            self.action_set = env.observe_actions()

            context = env.generate_context()
            a_index, action, pred_reward = self.select_action(context)
            features = self.feature_map(action, context)

            reward = env.reveal_reward(features)
            self.oracle.update(features, pred_reward, reward)

            env.record_regret(reward, [self.feature_map(a, context) for a in self.action_set])

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

        best_action_idx = np.argmax(rewards)
        best_reward = rewards[best_action_idx]

        probabilities = np.zeros(len(self.action_set))
        gaps = best_reward - rewards

        for i in range(len(self.action_set)):
            if i != best_action_idx:
                gap = max(0, gaps[i])
                probabilities[i] = 1.0 / (self.mu + self.learn_rate * gap)

        if np.sum(probabilities) > 0:
            probabilities = (probabilities / np.sum(probabilities)) * self.alpha * (((self.T - self.t) / self.T) ** self.t)

        probabilities[best_action_idx] = 1.0 - np.sum(probabilities)

        probabilities = np.clip(probabilities, 0, 1000)
        probabilities = probabilities / np.sum(probabilities)

        index = np.random.choice(np.arange(len(self.action_set)), p=probabilities)
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