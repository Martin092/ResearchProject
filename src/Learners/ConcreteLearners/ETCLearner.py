from ..AbstractLearner import AbstractLearner
from src.Environments import LinearEnvironment

class ETCLearner(AbstractLearner):

    def __init__(self, T : int, params : dict):
        super().__init__(T, params)

        self.optimal_action = None
        self.action_set = None
        self.d = None
        self.k = None


    def run(self, env : LinearEnvironment):

        self.action_set = env.observe_actions()
        self.d = env.get_ambient_dim()
        self.k = env.k

        for t in range(self.T):

            context = env.generate_context()
            action = self.select_action(context)
            features = self.feature_map(action, context)

            reward = env.reveal_reward(features)

            env.record_regret(reward, [self.feature_map(a, context) for a in self.action_set])
            self.history.append((action, context, reward))
            self.t += 1

        return env.get_cumulative_regret()


    def feature_map(self, action, context):
        return action


    def select_action(self, context):
        if self.t < self.k:
            return self.action_set[self.t]

        if self.t == self.k:
            best_reward = -float('inf')
            for (a, c, r) in self.history:
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