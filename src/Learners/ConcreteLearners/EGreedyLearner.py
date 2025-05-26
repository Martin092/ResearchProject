from src.Environments import AbstractEnvironment
from src.Learners import AbstractLearner
from sklearn.linear_model import SGDRegressor
import numpy as np

class EGreedyLearner(AbstractLearner):
    """
    Epsilon-greedy learner for a sparse linear bandit: chooses a random 
    action with probability epsilon, otherwise selects the action with 
    the highest estimated reward based on a sparse, linear model.

    Attributes:
        T: Horizon
        t: Current round
        history: A list of (action, reward) tuples, updated per round

        epsilon:
        action_set:
        regressor:
    """
    def __init__(self, T: int, params: dict):
        super().__init__(T, params)

        self.epsilon = params["epsilon"]
        assert 0 <= self.epsilon <= 1

        self.action_set = []
        self.regressor = None

    def run(self, env: AbstractEnvironment, logger = None):
        self.regressor = SGDRegressor(penalty="l1", alpha=0.01)
        for t in range(1, self.T + 1):
            self.action_set = env.observe_actions()

            context = env.generate_context()
            action = self.select_action(context)
            feature = self.feature_map(action, context)

            reward = env.reveal_reward(feature)

            self.regressor.partial_fit(feature.reshape(1, -1), [reward])
            self.history.append((action, context, reward))

            env.record_regret(reward, [self.feature_map(a, context) for a in self.action_set])

            if logger is not None:
                logger.log(t, reward, env.regret[-1])


    def select_action(self, context):
        if not hasattr(self.regressor, 'coef_') or np.random.rand() < self.epsilon:
            i = np.random.randint(len(self.action_set))
            return self.action_set[i]
        else:
            features = [self.feature_map(a, context) for a in self.action_set]
            estimated_rewards = self.regressor.predict(np.array(features))
            best_reward_id = np.argmax(estimated_rewards)
            return self.action_set[best_reward_id]
    
    def total_reward(self):
        total = 0
        for (_, _, reward) in self.history:
            total += reward
        return total
    
    def cum_reward(self):
        total = []
        for (a, c, r) in self.history:
            total.append(r)
        return total

    
