from abc import ABC, abstractmethod

class AbstractEnvironment(ABC):

    def __init__(self, params : dict):

        assert params["d"] is not None

        self.d = params["d"]

        self.action_set = []
        self.context_set = []
        self.curr_context = None

        self.regret = []
        self.cum_regret = 0.0

    @abstractmethod
    def reveal_reward(self, action):
        pass

    @abstractmethod
    def generate_context(self):
        pass

    @abstractmethod
    def record_regret(self, reward, feature_set):
        pass

    def observe_actions(self):
        return self.action_set

    def get_ambient_dim(self):
        return self.d

    def get_cumulative_regret(self):
        return self.cum_regret