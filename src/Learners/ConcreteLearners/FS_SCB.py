from src.Environments import AbstractEnvironment
from src.Learners import AbstractLearner
from sklearn.linear_model import SGDRegressor
import numpy as np


class FSSquareCB(AbstractLearner):
    def __init__(self, T, params):
        super().__init__(T, params)
        self.d = None
        self.models = None
        self.oracles = []
        self.l_rate = 1  # TODO make this not static


    def random_model_generator(self):
        pass

    def setup(self, env : AbstractEnvironment):
        self.d = env.get_ambient_dim()
        self.k = env.k
        self.models = np.ones((1, self.d), dtype=np.bool)
        for i in range(len(self.models)):
            self.oracles.append(SGDRegressor(penalty="l1", alpha=1))
        self.oracle_weights = np.ones(len(self.models))
        assert len(self.oracles) == len(self.models)
        assert len(self.oracle_weights) == len(self.models)

    def run(self, env : AbstractEnvironment, logger = None):
        self.setup(env)

        for t in range(1, self.T + 1):
            self.action_set = env.observe_actions()

            context = env.generate_context()
            a_index, action, pred_reward = self.select_action(context)
            features = self.feature_map(action, context)

            reward = env.reveal_reward(features)
            # TODO update oracles
            self.update_oracles(a_index, context, reward)

            env.record_regret(reward, [self.feature_map(a, context) for a in self.action_set])

            # Log the actions
            if logger is not None:
                logger.log_full(t, context, a_index, action, reward, env.regret[-1])

            self.history.append((action, context, reward))
            self.t += 1

        return env.get_cumulative_regret()


    def select_action(self, context):
        if self.t == 0:
            return 0, self.feature_map(self.action_set[0], context), 0

        rewards = np.zeros(len(self.action_set))
        for i, a in enumerate(self.action_set):
            feat = self.feature_map(a, context)
            expert_rewards = np.zeros(len(self.models))

            for i, M in enumerate(self.models):
                feat_subset = np.copy(feat)
                feat_subset[~M] = 0
                expert_rewards[i] = self.oracles[i].predict([feat_subset])

            y = self.aggregate_rewards(expert_rewards)
            rewards[i] = y

        probabilities = np.zeros(len(self.action_set))
        a_max = np.argmax(rewards)
        for i, r in enumerate(rewards):
            if i == a_max: continue
            probabilities[i] = 1/ (self.k + self.l_rate * (r - rewards[a_max]))
        probabilities[a_max] = 1 - np.sum(probabilities)
        index = np.random.choice(np.arange(self.k), size=1, p=probabilities)[0]

        return index, self.action_set[index], rewards[index]

    def update_oracles(self, action_index, context, real_reward):
        feat = self.feature_map(self.action_set[action_index], context)
        for i, oracle in self.oracles:
            feat_subset = np.copy(feat)
            feat_subset[~self.models[i]] = 0
            oracle.partial_fit(feat_subset.reshape(1, -1), real_reward)


    def aggregate_rewards(self, rewards, beta=-1, alpha=2):
        r_min = beta
        r_max = beta + alpha

        # TODO this will be called for each action, maybe it removes too many models
        good_models = np.where(r_min <= rewards <= r_max)
        self.models = self.models[good_models]
        self.oracle_weights = self.oracle_weights[good_models]
        rewards = (rewards[good_models] - beta) / alpha

        w_sum = np.sum(self.oracle_weights)
        vs = self.oracle_weights / w_sum

        delta0 = 0
        delta1 = 0

        for i, v in enumerate(vs):
            delta0 += v * np.exp(-2 * (rewards[i] ** 2))
            delta1 += v * np.exp(-2 * ((1 - rewards[i]) ** 2))

        delta0 = -0.5 * np.log(delta0)
        delta1 = -0.5 * np.log(delta1)

        pred_min = 1 - np.sqrt(delta1)
        pred_max = np.sqrt(delta0)

        y_final = np.random.uniform(pred_min, pred_max)
        assert pred_min <= y_final <= pred_max
        return y_final

    def feature_map(self, action, context):
        return (np.array(action) + context).reshape(-1, 1)

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