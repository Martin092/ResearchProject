import time

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from src.OnlineRegressors.AbstractRegressor import Regressor
from src.OnlineRegressors.ConcreteRegressors.LinearRegression import LinearRegression
from src.OnlineRegressors.ConcreteRegressors.RidgeRegression import OnlineRidge
from src.OnlineRegressors.ConcreteRegressors.AdaptiveRegression import AdaptiveRegressor
matplotlib.use("tkagg")
from tqdm import tqdm


def test(regressor: Regressor, d: int, n: int, train=0.7, sigma=0.1):
    data = sp.csr_matrix(sp.random(n, d, 0.2) * 10)
    w_star = np.random.rand(d, 1) * 10  # True weights
    y = data @ w_star
    y_noise = y + np.random.normal(0, sigma, size=(n, 1))

    # Split into training and testing sets
    n_train = int(train * n)
    data_train = data[:n_train, :]
    data_test = data[n_train:, :]
    y_test = y[n_train:]

    # Train the regressor
    for i in range(n_train):
        x = data_train[i].T
        p = regressor.predict(x)
        regressor.update(x, p, y_noise[i])

    # Test the regressor
    pred = np.empty(n - n_train)
    for i in range(n - n_train):
        x = data_test[i].T
        pred[i] = regressor.predict(x)
        regressor.update(x, pred[i], y_noise[i + n_train])

    mse = np.mean((pred - y_test.flatten()) ** 2)
    print(f"Final MSE: {mse}")
    print(f"Weight Error: {np.linalg.norm(regressor.w - w_star)}")

    plt.plot(np.abs(regressor.pred_history[400:] - regressor.real_history[400:]))
    plt.show()
    # MSE over time
    cumulative_mse = np.cumsum((regressor.pred_history - y.flatten()) ** 2) / np.arange(1, len(regressor.pred_history) + 1)
    plt.plot(cumulative_mse)
    plt.title("Average MSE")
    plt.show()

def test_adaptive(reg: Regressor, d, density, noise, n, eps=0.2, k=1, w_star=None):
    if w_star == None:
        w_star = sp.random(d, 1, density=density)
        w_star /= sp.linalg.norm(w_star, 1)

    reals = np.zeros(n)

    start = time.time()
    for i in tqdm(range(n)):
        xt = np.random.normal(0, 0.6, size=(d, 1))
        # xt = np.random.random((d, 1))
        xt_noisy = xt + np.random.normal(0, noise)
        y_pred = reg.predict(xt_noisy)
        y_real = w_star.T @ xt + np.random.normal(0, noise)
        reals[i] = w_star.T @ xt
        reg.update(xt_noisy, y_pred, y_real)
    runtime = time.time() - start

    mse = np.mean((reg.pred_history - reals) ** 2)
    print()
    print(f"Final MSE: {mse}")
    print(f"Weight Error: {np.linalg.norm(reg.w - w_star)}")

    # plt.plot(np.abs(reg.pred_history - reg.real_history))
    # plt.show()
    # MSE over time
    cumulative_mse = np.cumsum((reg.pred_history - reals) ** 2) / np.arange(1, len(reg.pred_history) + 1)
    return cumulative_mse, runtime, reals

d = 100
density = 0.15
noise = 1

w_star = sp.random(d, 1, density=density)
w_star /= sp.linalg.norm(w_star, 1)



eps=0.2
k=1
n = int((1 / (eps * eps)) * k * np.log(eps * d / k)) + 2
n *= 2
# n=1000
print(n)
t0 = k * np.log(d) * np.log(n)
params = {"sigma": noise,
          "k": int(density * d),
          "k0": int((density+0.05) * d),
          "t0": t0
    }

reg1 = AdaptiveRegressor(d, params)
reg2 = OnlineRidge(d)

res1, t1, reals1 = test_adaptive(reg1, d, density, noise, n, w_star=w_star)
res2, t2, reals2 = test_adaptive(reg2, d, density, noise, n, w_star=w_star)

print(f"Adaptive runtime: {t1}")
print(f"Ridge runtime: {t2}")

plt.plot(res1, label="POSLR")
plt.plot(res2, label="Ridge")
plt.legend()
plt.xlabel("Rounds")
plt.ylabel("MSE")
plt.title("Average MSE")
plt.show()


plt.plot(reg1.regret(w_star, reals=reals1), label="POSLR")
plt.plot(reg2.regret(w_star), label="Ridge")
plt.legend()
plt.xlabel("Rounds")
plt.ylabel("Regret")
plt.title("Average Regret")
plt.show()


# print(w_star.toarray().flatten())
# print(reg1.w.flatten())
# plt.scatter(w_star.toarray().flatten(), reg1.w.flatten(), label="POSLR", alpha=0.3)
# plt.scatter(w_star.toarray().flatten(), reg2.w.flatten(), label="Ridge", alpha=0.3)
# plt.title("Weights")
# plt.xlabel("Real value")
# plt.ylabel("Predicted value")
# plt.legend()
# plt.show()

