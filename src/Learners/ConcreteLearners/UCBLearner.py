from src.Environments import AbstractEnvironment
from src.Learners import AbstractLearner

import numpy as np


class UCBLearner(AbstractLearner):

    def __init__(self, T : int, params : dict):
        super().__init__(T, params)

        self.delta = params["delta"]

        assert 0 < self.delta <= 1

        self.counts = []
        self.averages = []
        self.action_set = []
        self.rewards = []


    def run(self, env: AbstractEnvironment, logger):


        self.action_set = env.observe_actions()  # Actions should be unique.
        self.counts = [0] * len(self.action_set)
        self.averages = [0] * len(self.action_set)

        for t in range(1, self.T + 1):

            # Reveal context, and select action (obliviously)
            context = env.generate_context()
            action = self.select_action(context)

            # Reveal the reward
            reward = env.reveal_reward(np.array(action).reshape(1, -1))
            self.rewards.append(reward)

            # Update counts and reward averages
            action_index = self.action_set.index(action)

            self.averages[action_index] = (self.averages[action_index] * self.counts[action_index] + reward) / (1 + self.counts[action_index])
            self.counts[action_index] += 1


            # Record the regret, and log
            env.record_regret(reward, self.action_set)
            if logger is not None:
                logger.log(t, reward, env.regret[-1])


    def select_action(self, context):
        # Compute the index function for all actions,
        # and select the action with the highest index
        best_ucb = np.argmax([self.compute_ucb_index(a) for a in self.action_set])
        action = self.action_set[best_ucb]

        return action

    def compute_ucb_index(self, action):

        index = self.action_set.index(action)

        if self.counts[index] == 0:
            return float('inf')

        ucb = self.averages[index] + np.sqrt(2 * np.log(1 / self.delta)/ self.counts[index])

        return ucb


    def total_reward(self):

        return np.sum(self.rewards)

    def cum_reward(self):
        return np.cumsum(self.rewards)