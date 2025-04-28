from abc import ABC, abstractmethod
from src.Environments import AbstractEnvironment

class AbstractLearner(ABC):
    def __init__(self, T : int, params : dict):
        '''
        :param actions: Array of the actions of the agent
        :param T: Horizon
        '''
        self.T = T
        self.t = 0
        self.history = []

    @abstractmethod
    def run(self, env : AbstractEnvironment):
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
        return action