from abc import ABC, abstractmethod
from src.Environments import AbstractEnvironment
import numpy as np

class AbstractLearner(ABC):
    def __init__(self, T : int, params : dict):
        '''
        :param actions: Array of the actions of the agent
        :param T: Horizon
        '''
        self.T = T
        self.t = 0
        self.history = []
        self.features_subset = None
        if "features_subset" in params.keys():
            self.features_subset = params["features_subset"]

    @abstractmethod
    def run(self, env : AbstractEnvironment, logger):
        pass

    @abstractmethod
    def select_action(self, context):
        pass

    @abstractmethod
    def total_reward(self):
        pass

    @abstractmethod
    def cum_reward(self):
        pass

    def feature_map(self, action, context):
        d = len(action)
        if self.features_subset:
            mask = np.random.choice(d, size=d - self.features_subset, replace=False)
            action[mask] = 0
        return action