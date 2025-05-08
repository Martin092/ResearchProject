import numpy as np
from sklearn.linear_model import SGDRegressor

from src.OnlineRegressors.AbstractRegressor import Regressor


class SGDWrapper(Regressor):
    def __init__(self, d):
        super().__init__(d)
        self.reg = SGDRegressor(penalty='l2', alpha=0.01)
        self.t = 0

    def predict(self, x):
        if self.t == 0:
            return 0
        return self.reg.predict(x.reshape(1, -1))

    def update(self, x, pred, real):
        self.reg.partial_fit(x.reshape(1, -1), [real])
        self.t += 1

    def regret(self, w_optimal, reals=None):
        return 0
