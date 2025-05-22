from src.OnlineRegressors.AbstractRegressor import Regressor
import numpy as np

class RidgeFSSCB(Regressor):
    def __init__(self, d, params):
        super().__init__(d, params)
        self.x_history = np.empty((0, d))

        self.lambda_reg = params["lambda_reg"]
        self.w = np.zeros(d)
        self.lam_eye = self.lambda_reg * np.eye(self.d)
        self.V = np.copy(self.lam_eye)

    def predict(self, x):
        assert x.ndim == 1
        return np.dot(self.w, x)

    def update(self, x, pred, real):
        assert x.ndim == 1
        assert pred.ndim == 0
        assert real.ndim == 0

        self.x_history = np.vstack((self.x_history, x))
        self.real_history = np.append(self.real_history, real)

        self.V = self.lam_eye + self.x_history.T @ self.x_history
        self.w = np.linalg.inv(self.V) @ self.x_history.T @ self.real_history

        self.pred_history = np.append(self.pred_history, pred)

    def regret(self, w_optimal, reals=None):
        loss = 0
        optimal_loss = 0
        regrets = np.array([])
        for i, pred in enumerate(self.pred_history):
            loss += (pred - self.real_history[i]) ** 2
            optimal_loss += ((w_optimal @ self.x_history[i]) - self.real_history[i]) ** 2
            regrets = np.append(regrets, loss - optimal_loss)

        return regrets
