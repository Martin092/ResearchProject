from abc import ABC,  abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp


class Regressor(ABC):
    def __init__(self, d, params):
        self.x_history = np.empty((0, d))
        self.pred_history = np.array([])
        self.real_history = np.array([])
        self.wt = np.array([np.zeros(d)])
        self.d = d

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def update(self, x, pred, real):
        pass

    @abstractmethod
    def regret(self, w_optimal, reals=None):
        pass

    def predict_and_fit(self, x, y):
        pred = self.predict(x)
        self.update(x, pred, y)
        return pred