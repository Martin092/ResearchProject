from src.OnlineRegressors.AbstractRegressor import Regressor
import numpy as np

class LinearRegression(Regressor):
    def __init__(self, d, eta):
        super().__init__(d)
        self.eta = eta
        self.w = np.zeros((d, 1))

    def predict(self, x):
        pred = self.w.T @ x
        # self.x_history = np.vstack((self.x_history, x))
        # self.pred_history = np.append(self.pred_history, pred)
        return pred

    def update(self, x, pred, real):
        self.w = self.w - self.eta * (pred - real) * x
        self.real_history = np.append(self.real_history, real)
        # self.wt = np.vstack((self.wt, w_new))

    def regret(self, w_optimal, reals=None):
        loss = 0
        optimal_loss = 0
        regrets = np.array([])
        for i, pred in enumerate(self.pred_history):
            loss += (pred - self.real_history[i]) ** 2
            optimal_loss += (np.dot(w_optimal, self.x_history[i]) - self.real_history[i]) ** 2
            regrets = np.append(regrets, loss - optimal_loss)

        return regrets