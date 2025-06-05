from src.OnlineRegressors.AbstractRegressor import Regressor
import numpy as np

class BayesianSelection(Regressor):
    def __init__(self, d, params):
        super().__init__(d, params)
        self.x_history = np.empty((0, d))

        self.lambda_reg = params["lambda_reg"]
        self.k = params["k"]
        self.full_x = params.get("full_x", True)
        self.debias = params.get("debias", False)

        self.rho = 1e-4
        self.c = 100000
        self.noise = 0.01

        self.gammas_probs = np.repeat(0.5, self.d)
        self.gammas = self.sample_gammas(self.gammas_probs)
        self.w = np.zeros(d)
        self.integral = np.zeros(self.d)
        self.mask = np.ones(self.d)

    def sample_gammas(self, gammas_probs):
        return [1 if np.random.rand() < p else 0 for p in gammas_probs]

    def var_w(self, gamma):  # variance of w
        assert gamma in [0, 1]
        return self.rho if gamma == 0 else self.c * self.rho

    def sample_weights(self, gammas):
        return [np.random.normal(0, self.var_w(gamma)) for gamma in gammas]

    def gaussian(self, x, mu, var):
        coef = 1 / np.sqrt(var * 2 * np.pi)
        exp = np.exp(-0.5 * ((x - mu) ** 2) / var)
        return coef * exp

    def prob_w(self, w, gamma):
        return self.gaussian(w, 0, self.var_w(gamma))

    def prob_gamma(self, gamma):
        return self.rho if gamma == 0 else self.c * self.rho

    def gamma_post(self, k, gammas_probs, gammas, ws):
        p0 = (1 - gammas_probs[k]) * self.prob_w(ws[k], 0)
        p1 = gammas_probs[k] * self.prob_w(ws[k], 1)
        return p1 / (p0 + p1)

    def w_post(self, X, y, gammas, noise):  # p(w | stuff)
        diag = [1 / ((self.c ** gamma) * self.rho) for gamma in gammas]
        R_gamma_inv = np.diag(diag)
        xtx = X.T @ X
        mean = np.linalg.inv(noise * R_gamma_inv + xtx) @ X.T @ y
        var = np.linalg.inv(R_gamma_inv + xtx / noise)

        return np.random.multivariate_normal(mean, var)

    def predict(self, x):
        assert x.ndim == 1
        return np.dot(np.multiply(self.w, self.mask), x)

    def update(self, x, pred, real):
        assert x.ndim == 1
        assert pred.ndim == 0
        assert real.ndim == 0

        self.w = self.w_post(self.x_history, self.real_history, self.gammas, self.noise)
        for i in range(len(self.gammas_probs)):
            self.gammas_probs[i] = self.gamma_post(i, self.gammas_probs, self.gammas, self.w)
        self.gammas = self.sample_gammas(self.gammas_probs)

        w_probs = np.zeros(self.d)
        for i in range(len(w_probs)):
            w_probs[i] = self.prob_w(self.w[i], self.gammas[i])
        self.integral += self.w * self.gammas * w_probs * self.gammas_probs
        self.mask = np.zeros(self.d)
        self.mask[np.argsort(-self.integral)[:self.k]] = 1

        if not self.full_x:
            x = np.multiply(x, self.mask)

        if self.debias:
            integral = np.clip(self.integral, 0, np.inf)
            integral /= np.sum(integral)
            inv = 1/integral
            inv[np.isinf(inv)] = 0
            x = np.multiply(inv, x)

        self.real_history = np.append(self.real_history, real)
        self.x_history = np.vstack((self.x_history, x))
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
