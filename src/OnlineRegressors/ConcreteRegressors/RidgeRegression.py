from src.OnlineRegressors.AbstractRegressor import Regressor
import numpy as np

class OnlineRidge(Regressor):
    def __init__(self, d, params):
        super().__init__(d, params)
        self.x_history = []

        self.lambda_reg = params["lambda_reg"]
        self.w = np.zeros((d, 1))
        self.wt = np.array([self.w])
        self.M = np.eye(d) / self.lambda_reg

    def predict(self, x):
        # dot product
        return (self.w.T @ x.reshape(-1, 1))

    def update(self, x, pred, real):
        Mx = self.M @ x
        denom = 1 + x.T @ Mx

        self.w += (self.M / denom) @ (x * (real - pred))
        self.M = self.M - np.outer(Mx, Mx) / denom

        self.x_history.append([x])
        self.wt = np.append(self.wt, [self.w], axis=1)
        self.pred_history = np.append(self.pred_history, pred)
        self.real_history = np.append(self.real_history, real)

    def regret(self, w_optimal):
        loss = 0
        optimal_loss = 0
        regrets = np.array([])
        for i, pred in enumerate(self.pred_history):
            loss += (pred - self.real_history[i]) ** 2
            optimal_loss += ((w_optimal.T @ self.x_history[i][0]) - self.real_history[i]) ** 2
            regrets = np.append(regrets, loss - optimal_loss)

        return regrets
