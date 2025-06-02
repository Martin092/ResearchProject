import numpy as np

def sample_gammas(gammas_probs):
    return [1 if np.random.rand() < p else 0 for p in gammas_probs]

def var_w(gamma, c, rho): # variance of w
    assert gamma in [0, 1]
    return rho if gamma == 0 else c * rho

def sample_weights(gammas, c, rho):
    return [np.random.normal(0, var_w(gamma, c, rho)) for gamma in gammas]

def gaussian(x, mu, var):
    coef = 1 / np.sqrt(var * 2 * np.pi)
    exp = np.exp(-0.5 * ((x - mu) ** 2)/var)
    return coef * exp

def prob_w(w, gamma, c, rho):
    return gaussian(w, 0, var_w(gamma, c, rho))

def prob_gamma(gamma, c, rho):
    return var_w(gamma, c, rho)

def gamma_post(k, gammas_probs, ws, c, rho):
    p0 = (1 - gammas_probs[k]) * prob_w(ws[k], 0, c, rho)
    p1 = gammas_probs[k] * prob_w(ws[k], 1, c, rho)
    return p1 / (p0 + p1)

def w_post(X, y, gammas, noise, c, rho): # p(w | stuff)
    diag = [1 / ((c ** gamma) * rho) for gamma in gammas]
    R_gamma_inv = np.diag(diag)
    xtx = X.T @ X
    mean = np.linalg.inv(noise * R_gamma_inv + xtx) @ X.T @ y
    var = np.linalg.inv(R_gamma_inv + xtx / noise)

    return np.random.multivariate_normal(mean, var)


def gibbs(X, y, n, noise, c, rho):
    gammas_probs = np.repeat(0.1, X.shape[1])
    gammas = sample_gammas(gammas_probs)
    ws = w_post(X, y, gammas, noise, c, rho)

    integral = np.zeros(X.shape[1])
    for i in range(n):
        ws = w_post(X, y, gammas, noise, c, rho)
        for i in range(len(gammas_probs)):
            gammas_probs[i] = gamma_post(i, gammas_probs, ws, c, rho)
        gammas = sample_gammas(gammas_probs)

        w_probs = np.zeros(X.shape[1])
        gamma_prior = np.zeros(X.shape[1])
        for i in range(len(w_probs)):
            w_probs[i] = prob_w(ws[i], gammas[i], c, rho)
            gamma_prior[i] = prob_gamma(gammas[i], c, rho)

        integral += ws * gammas * w_probs * gammas_probs
    return ws, gammas, integral