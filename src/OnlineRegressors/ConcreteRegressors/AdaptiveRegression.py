import numpy

from src.OnlineRegressors.AbstractRegressor import Regressor
import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp
import cvxpy as cp

class AdaptiveRegressor(Regressor):
    def __init__(self, d, params):
        super().__init__(d)
        self.x_history = []
        self.sigma = params["sigma"]
        # sparsity of regressors we compare the algorithms performance to
        self.k = params["k"]
        # number of features we can query each round
        self.k0 = params["k0"]
        self.t0 = params["t0"]
        self.t = 1
        self.w = np.zeros(self.d)
        self.Xt = np.array([])

    def predict(self, x):
        if self.t <= self.t0:
            self.t += 1
            return self.w.T @ x
        elif np.log2(self.t) == int(np.log2(self.t)):
            # TODO Read what they do here, this is wrong
            w_hat = self.dantzig_selector().flatten()
            sorted_args = np.argsort(np.abs(w_hat))
            w_hat[sorted_args[self.d - self.k:]] = 0
            self.w = w_hat.reshape(-1, 1)
            self.t += 1
            return self.w.T @ x
        else:
            self.t += 1
            return self.w.T @ x


    def update(self, x, pred, real):
        St = np.random.choice(np.arange(self.d), size=self.d - self.k0, replace=False)
        proj = x.copy()
        proj[St] = 0
        proj = (self.d / self.k0) * proj  # projection so set all other values to 0
        proj = sp.csr_matrix(proj)
        self.x_history.append(x)
        self.real_history = np.append(self.real_history, real)
        self.pred_history = np.append(self.pred_history, pred)

        if self.Xt.shape[0] == 0:
            self.Xt = proj.reshape(1, -1)
        else:
            self.Xt = sp.vstack([self.Xt, proj.reshape(1, -1)])

    def regret(self, w_optimal, reals=None):
        if reals is None:
            reals = self.real_history
        loss = 0
        optimal_loss = 0
        regrets = np.array([])
        for i, pred in enumerate(self.pred_history):
            loss += (pred - reals[i]) ** 2
            optimal_loss += ((w_optimal.T @ self.x_history[i]) - reals[i]) ** 2
            regrets = np.append(regrets, loss - optimal_loss)

        return regrets

    def dantzig_selector(self, delta=0.05, C=1):
        t_inv = 1/self.t
        a = t_inv * (self.Xt.T @ (self.real_history.reshape(-1, 1)))
        Dt_diag = (1 - self.k0 / self.d) * (self.Xt.T @ self.Xt).diagonal()
        Dt = sp.diags(Dt_diag)
        B = t_inv * (-self.Xt.T @ self.Xt + Dt)

        denominator = self.d * np.log(self.t * self.d / delta)
        lhs = C * np.sqrt(denominator/(self.t * self.k0)) * (self.sigma + self.d / self.k0)

        # maybe normalize
        w = cp.Variable((self.d, 1))
        constraints = [cp.norm(a + B @ w, 'inf') <= lhs]
        obj = cp.Minimize(cp.norm(w, 1))
        prob = cp.Problem(obj, constraints)
        result = prob.solve(verbose=False)
        return w.value
