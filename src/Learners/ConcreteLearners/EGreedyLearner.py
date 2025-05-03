from src.Environments import SparseLinearEnvironment
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
            context = env.generate_context() # why?
            action = self.select_action(context)

            x = np.array(action).reshape(1, -1)

            # Compute the reward corresponding to the selected action (feature vector)
            reward = env.reveal_reward(x)

            # Append (action, reward) at round t to history
            self.history.append(action, reward)

            # Update regressor
            self.regressor.partial_fit(np.array(action).reshape(1, -1), reward)

            if logger is not None:
                logger.log(t, reward, env.regret[-1])

    def select_action(self, context):

        # If regressor hasn't been run yet, then the parameter 'coef_' doesn't exist.
        if not hasattr(self.regressor, 'coef_') or np.random.rand() < self.epsilon:
            return np.random.rand(self.action_set)

        else:
            # Compute estimated rewards using estimated theta and feature vectors
            estimated_rewards = self.regressor.predict(self.action_set)
            
            # Select action index whose corresponding estimated reward is maximum. 
            best_idx = np.argmax(estimated_rewards)
            
            # Return feature vector corresponding to selected action. 
            return self.action_set[best_idx] 
    
    def total_reward(self):
        total = 0
        for (_, _, reward) in self.history:
            total += reward
        return total
    
    def cum_reward(self):
        cumulative = []
        for (_, _, reward) in self.history:
             cumulative.append[reward]
        return cumulative


    
