import numpy as np
import scipy.sparse as sp
import cvxpy as cp

from scipy.optimize import linprog
from src.OnlineRegressors.AbstractRegressor import Regressor


class AdaptiveRegressor(Regressor):
    def __init__(self, d, params):
        super().__init__(d, params)
        self.d = d
        self.sigma = params["sigma"]
        self.k = params["k"]
        self.k0 = params["k0"]
        self.t0 = params["t0"]
        self.C = params["C"]
        self.delta = params["delta"]

        self.t = 1
        self.w = np.zeros(d)
        self.x_history = []
        self.real_history = np.empty((0, 1))
        self.pred_history = np.empty(0)
        self.Xt = sp.csr_matrix((0, d))

    def predict(self, x):
        assert x.ndim == 1, f"x has shape {x.shape}"
        return np.dot(self.w, x)

    def update(self, x, pred, real):
        assert x.ndim == 1
        assert pred.ndim == 0
        assert real.ndim == 0
        # more than t0 and is power of 2
        if self.t > self.t0 and (self.t & (self.t - 1) == 0):
            w_hat = self.dantzig_selector().flatten()
            idx = np.argsort(np.abs(w_hat))
            w_hat[idx[:self.d - self.k]] = 0
            self.w = w_hat

        # Increment time
        self.t += 1

        # Random projection step
        mask = np.random.choice(self.d, size=self.d - self.k0, replace=False)
        proj = x.copy()
        proj[mask] = 0
        proj *= self.d / self.k0
        proj = sp.csr_matrix(proj)

        # Record histories
        self.x_history.append(x)
        self.pred_history = np.append(self.pred_history, pred)
        self.real_history = (
            np.array([[real]])
            if self.real_history.size == 0
            else np.vstack((self.real_history, [[real]]))
        )

        # Stack feature matrix
        self.Xt = proj if self.Xt.shape[0] == 0 else sp.vstack([self.Xt, proj])

    def regret(self, w_optimal, reals=None):
        assert w_optimal.shape[0] == 1
        assert w_optimal.shape[1] == self.d

        if reals is None:
            reals = self.real_history
        loss = 0
        optimal_loss = 0
        regrets = np.array([])
        for i, pred in enumerate(self.pred_history):
            loss += (pred - reals[i]) ** 2
            optimal_loss += ((w_optimal @ self.x_history[i]) - reals[i]) ** 2
            regrets = np.append(regrets, loss - optimal_loss)

        return regrets

    def dantzig_selector(self):
        t_inv = 1.0 / self.t
        # resid0 = t_inv * (self.Xt.T @ self.real_history)

        term = self.C * np.sqrt(self.d * np.log(self.t * self.d / self.delta) / (self.t * self.k0))
        rhs = term * (self.sigma + self.d / self.k0)
        # print("‖resid0‖∞ =", np.max(np.abs(resid0)), "rhs =", rhs)

        Dt_diag = (1 - self.k0 / self.d) * (self.Xt.T @ self.Xt).diagonal()
        Dt = sp.diags(Dt_diag)

        rhs = rhs * np.ones((self.d, 1))

        w = cp.Variable((self.d, 1))

        expr = t_inv * self.Xt.T @ (self.real_history - self.Xt @ w) + t_inv * Dt @ w
        constraints = [expr <= rhs, expr >= -rhs]

        prob = cp.Problem(cp.Minimize(cp.norm(w, 1)), constraints)
        prob.solve(solver=cp.ECOS, abstol=1e-8, reltol=1e-8, feastol=1e-8)

        return w.value
