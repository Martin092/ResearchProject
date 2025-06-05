from src.Environments import AbstractEnvironment
from src.Learners import AbstractLearner
from sklearn.linear_model import SGDRegressor
import numpy as np
from itertools import combinations
from math import comb

from src.OnlineRegressors.ConcreteRegressors.OnlineRidgeFSSCB import RidgeFSSCB
from sklearn.feature_selection import SelectKBest, f_regression
from src.utility.GibsSampler import gibbs
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

class FSSquareCB(AbstractLearner):
    def __init__(self, T, params):
        super().__init__(T, params)
        self.X = None
        self.d = None
        self.models = None
        self.oracles = []
        self.oracle_weights = None
        self.l_rate = params["l_rate"]
        self.num_features = params["num_features"]
        self.strategy = params["strategy"]
        self.strat_params = params["strat_params"]
        self.strat_map = {
            "random": self.random_model_generator,
            "all_subsets": self.all_subsets,
            "theta_weights": self.theta_weights, # M, warmup, s
            "theta_weights_exp": self.theta_weights,
            "AnovaF": self.theta_weights,
            "mcmc": self.subset_features,
            "bayesian": self.theta_weights
          }
        self.learn_rate = params["learn_rate"]

        self.delta = 0.1
        self.G = 1
        self.S = 1
        self.L = 1

    def theta_weights(self):
        return np.ones((1, self.d), dtype=bool)

    def subset_features(self):
        arr = np.zeros((self.strat_params["M"], self.d), dtype=bool)
        idx = np.arange(self.d)
        for m in range(self.strat_params["M"]):
            np.random.shuffle(idx)
            arr[m][idx[:self.num_features]] = 1
        return arr


    def theta_weights_warmed_up(self):
        assert self.strategy == "theta_weights_exp" or self.models.shape[0] == 1
        assert self.strategy == "theta_weights_exp" or len(self.oracles) == 1
        assert self.strategy == "theta_weights_exp" or len(self.oracle_weights) == 1

        M = self.strat_params["M"]
        s = self.strat_params["s"]

        theta = self.oracles[0].w
        indices = np.argsort(np.abs(theta.flatten()))[self.d - self.num_features - s:]
        self.sample_models(M, indices)

    def accept_prob(self, reward):
        return np.exp((1/np.log(self.t+1)) * reward)

    def mcmc_selection(self, x, y):
        for m in range(self.strat_params["M"]):
            S = self.models[m]
            i = np.random.randint(0, self.d)
            S_prime = S.copy()
            S_prime[i] = 1 - S_prime[i] # flip the bit

            y_new = self.oracles[m].predict(x)
            p_accept = self.accept_prob(y_new) / self.accept_prob(y)
            if np.random.uniform(0,1) < p_accept:
                # print(f"Accepted from {y} to {} ")
                self.models[m] = S_prime

    def bayesian_selection(self, env):
        assert self.strategy == "bayesian"
        assert self.models.shape[0] == 1
        assert len(self.oracles) == 1
        assert len(self.oracle_weights) == 1

        M = self.strat_params["M"]
        s = self.strat_params["s"]
        rho = 1e-4
        c = 10000
        noise = 0.01
        iterations = self.strat_params["gibbs_iterations"]

        rewards = list(map(lambda x: x[2], self.history))

        ws, gammas, integral = gibbs(self.X, rewards, iterations, noise, c, rho)
        indices = np.argsort(-integral)[:self.num_features + s]

        # integral = np.clip(integral, 0, np.inf)
        # print((integral / np.sum(integral))[indices][:10])
        # pred = np.sort(indices[:5])
        # real = np.sort(env.feat_mask()[1])
        # coincide = 0
        # for p in pred:
        #     if p in real:
        #         coincide += 1
        #
        # print(f"{coincide}/5 coincide")
        # print("Mean is ", np.sum((integral / np.sum(integral))[indices][:5]))
        # print(env.true_theta)
        #
        # plt.bar(np.arange(self.d), integral / np.sum(integral), label="p(gamma | ...)")
        # plt.scatter(real, np.zeros(5), label="Non zero weights")
        # plt.legend()
        # plt.show()

        self.models = np.zeros((M, self.d), dtype=np.bool)
        self.oracles = []
        self.oracle_weights = np.ones(len(self.models))

        self.models[0][indices[:self.num_features]] = True
        self.oracles.append(RidgeFSSCB(self.d, {"lambda_reg": self.l_rate}))

        for i in range(1, M):
            np.random.shuffle(indices)
            self.models[i][indices[:self.num_features]] = True
            self.oracles.append(RidgeFSSCB(self.d, {"lambda_reg": self.l_rate}))

        self.oracles = np.array(self.oracles)



    def anova_selection(self, env):
        assert self.strategy == "AnovaF"
        assert self.models.shape[0] == 1
        assert len(self.oracles) == 1
        assert len(self.oracle_weights) == 1

        M = self.strat_params["M"]
        s = self.strat_params["s"]

        selector = SelectKBest(f_regression, k=self.num_features + s).fit(self.X, list(map(lambda x: x[2], self.history)))
        indices = selector.get_support()
        indices = [i for i, val in enumerate(indices) if val]


        self.sample_models(M, indices)

    def sample_models(self, M, indices):
        '''
        Given a subset of all the features, it samples M models from the indices.
        Each of the models is a mask with features being a subset of indices
        It is therefore important to have the lenght of indices be longer
        than the length of the features you want to project
        :param M:
        :param indices:
        :return:
        '''
        assert len(indices) >= self.num_features

        self.models = np.zeros((M, self.d), dtype=np.bool)
        self.oracles = []
        self.oracle_weights = np.ones(len(self.models))
        for i in range(M):
            np.random.shuffle(indices)
            self.models[i][indices[:self.num_features]] = True
            self.oracles.append(RidgeFSSCB(self.d, {"lambda_reg": self.l_rate}))
            assert np.sum(self.models[i]) == self.num_features
        self.oracles = np.array(self.oracles)
        assert len(self.oracles) == len(self.models)
        assert len(self.oracle_weights) == len(self.models)

    def all_subsets(self):
        k = self.num_features
        n = self.d
        assert k <= n

        total = comb(n, k)
        result = np.zeros((total, n), dtype=bool)
        for i, idxs in enumerate(combinations(range(n), k)):
            result[i, list(idxs)] = True
        return result

    def random_model_generator(self):
        M = self.strat_params["M"]

        indices = np.arange(self.d)
        models = np.zeros((M, self.d), dtype=np.bool)
        for i in range(M):
            np.random.shuffle(indices)
            models[i][indices[:self.num_features]] = True
            assert np.sum(models[i]) == self.num_features

        return models


    def setup(self, env : AbstractEnvironment):
        self.d = env.get_ambient_dim()
        self.k = env.k
        self.X = np.empty((0, self.d))
        self.models = self.strat_map[self.strategy]()
        for i in range(len(self.models)):
            self.oracles.append(RidgeFSSCB(self.d, {"lambda_reg": self.l_rate}))
        self.oracles = np.array(self.oracles)
        self.oracle_weights = np.ones(len(self.models))
        assert len(self.oracles) == len(self.models)
        assert len(self.oracle_weights) == len(self.models)

    def run(self, env : AbstractEnvironment, logger = None):
        self.setup(env)

        for t in range(1, self.T + 1):
            if self.strategy == "theta_weights" and self.strat_params["warmup"] == t:
                self.theta_weights_warmed_up()
            elif self.strategy == "theta_weights_exp" and self.t & (self.t - 1) == 0:
                self.theta_weights_warmed_up()
            elif self.strategy == "AnovaF" and self.strat_params["warmup"] == t:
                self.anova_selection(env)
            elif self.strategy == "bayesian" and self.strat_params["warmup"] == t:
                self.bayesian_selection(env)

            self.action_set = env.observe_actions()

            context = env.generate_context()
            a_index, action, pred_reward = self.select_action(context)
            features = self.feature_map(action, context)
            self.X = np.vstack((self.X, features))

            reward = env.reveal_reward(features)
            self.update_oracles(a_index, context, reward)

            if self.strategy == "mcmc":
                self.mcmc_selection(features, reward)

            env.record_regret(reward, [self.feature_map(a, context) for a in self.action_set])

            # Log the actions
            if logger is not None:
                logger.log_full(t, context, a_index, action, reward, env.regret[-1])

            # self.history.append((action, context, reward))
            self.history.append((None, None, reward))
            self.t += 1

        return env.get_cumulative_regret()


    def select_action(self, context):
        if self.t == 0:
            return 0, self.feature_map(self.action_set[0], context), 0

        rewards = np.zeros(len(self.action_set))
        for i, a in enumerate(self.action_set):
            feat = self.feature_map(a, context)
            expert_rewards = np.zeros(len(self.models))

            for j, M in enumerate(self.models):
                feat_subset = np.copy(feat)
                feat_subset[~M] = 0
                expert_rewards[j] = self.oracles[j].predict(feat_subset)

            y = self.aggregate_rewards(expert_rewards)
            rewards[i] = y

        best_action_idx = np.argmax(rewards)
        best_reward = rewards[best_action_idx]
        gaps = rewards - best_reward

        probabilities = np.zeros(len(self.action_set))

        for i in range(len(self.action_set)):
            if i != best_action_idx:
                probabilities[i] = 1.0 / (self.k + self.learn_rate * gaps[i])

        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)

        probabilities[best_action_idx] = 1.0 - np.sum(probabilities)

        probabilities = np.clip(probabilities, 0, 1)
        probabilities = probabilities / np.sum(probabilities)

        index = np.random.choice(np.arange(self.k), size=1, p=probabilities)[0]
        return index, self.action_set[index], rewards[index]

    def update_oracles(self, action_index, context, real_reward):
        feat = self.feature_map(self.action_set[action_index], context)
        r_max = self.rmax()
        r_min = -r_max
        real_scaled = (real_reward - r_min) / (r_max - r_min)
        for i, oracle in enumerate(self.oracles):
            feat_subset = np.copy(feat)
            feat_subset[~self.models[i]] = 0
            pred = oracle.predict(feat_subset)
            oracle.update(feat_subset, pred, real_reward)

            # TODO Weights use a scaled predictions, this is possibly bad. Not sure
            self.oracle_weights[i] *= np.exp(-2 * (real_scaled - (pred - r_min) / (r_max - r_min))**2)

    def rmax(self):
        delta = self.delta
        G = self.G
        S = self.S
        L = self.L
        return G + L * np.sqrt(self.d * np.log((1 + self.t * (L ** 2) / (self.l_rate * self.d)) / delta)) + L * S * np.sqrt(self.l_rate)

    def aggregate_rewards(self, rewards):
        r_max = self.rmax()
        r_min = -r_max

        # print(r_max)
        # print(max(rewards))
        # print(min(rewards))
        # print("----------------------")

        # TODO this will be called for each action, maybe it removes too many models
        good_models = (r_min <= rewards) & (rewards <= r_max)
        if np.sum(good_models) != len(self.models):
            print("Removed ", len(self.models) - np.sum(good_models))
            print()

        self.models = self.models[good_models]
        self.oracle_weights = self.oracle_weights[good_models]
        self.oracles = self.oracles[good_models]
        hs = (rewards[good_models] - r_min) / ((r_max - r_min))

        w_sum = np.sum(self.oracle_weights)
        vs = self.oracle_weights / w_sum

        delta0 = 0
        delta1 = 0

        for i, v in enumerate(vs):
            delta0 += v * np.exp(-2 * (hs[i] ** 2))
            delta1 += v * np.exp(-2 * ((1 - hs[i]) ** 2))

        delta0 = -0.5 * np.log(delta0)
        delta1 = -0.5 * np.log(delta1)

        pred_min = 1 - np.sqrt(delta1)
        pred_max = np.sqrt(delta0)

        y_final = (pred_max + pred_min) / 2.0
        # print(y_final)
        # print(pred_max)
        # print(pred_min)
        # print()
        assert pred_min - 1e-6 <= y_final <= pred_max + 1e-6
        return y_final * (r_max - r_min) + r_min

    def feature_map(self, action, context):
        return (action)

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