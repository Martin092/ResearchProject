from src.Environments import SparseLinearEnvironment
from src.Learners import AbstractLearner
from sklearn.linear_model import SGDRegressor
import numpy as np

class EGreedyLearner(AbstractLearner):
    """
    Epsilon-greedy learner for a sparse linear bandit:
    chooses a random action with probability epsilon, otherwise selects the 
    action with the highest estimated reward based on a sparse, linear model.

    Attributes:
        T: Horizon
        t: Current round
        history: A list of (action, reward) tuples, updated per round

        epsilon:
    """
    def __init__(self, T: int, params: dict):
        super().__init__(T, params)

        self.epsilon = params["epsilon"]
        assert 0 <= self.epsilon <= 1

        self.counts = []
        self.averages = []
        self.action_set = []
        self.rewards = []

        # Lasso regressor to compute estimated theta (parameter vector)
        self.regressor = SGDRegressor(penalty="l1", alpha=0.01)

        # Elastic net regressor
        # By default, l1_ratio = 0.15, meaning l1=0.0015 and l2=0.0085
        ## self.regressor = SGDRegressor(penalty="elasticnet", alpha=0.01)

    def run(self, env: SparseLinearEnvironment, logger = None):

        for t in range(1, self.T + 1):
            
            # Generate a new set of actions (feature vectors)
            self.action_set = env.observe_actions()

            # Select an action (feature vector) through exploration or exploitation
            action = self.select_action()

            # Compute the reward corresponding to the selected action (feature vector)
            reward = env.reveal_reward(action)

            # Append (action, reward) at round t to history
            self.history.append(action, reward)

            if logger is not None:
                logger.log(t, reward, env.record_regret[-1])

            self.history.append(action, reward)
            

        # action count
        self.counts = [0] * len(self.action_set)


    def select_action(self):

        if not hasattr(self.regressor, 'coef_') or np.random.rand() < self.epsilon:
            return np.random.rand(self.action_set)

        else:
            preds = self.regressor.predict(self.action_get)
            return np.argmax(preds)


    def total_reward(self):
        return super().total_reward()
    
    def cum_reward(self):
        return super().cum_reward()
    
    def feature_map(self, action, context):
        return np.array(action)


    
