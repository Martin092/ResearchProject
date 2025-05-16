import time

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from src.OnlineRegressors.AbstractRegressor import Regressor
from src.OnlineRegressors.ConcreteRegressors.LinearRegression import LinearRegression
from src.OnlineRegressors.ConcreteRegressors.RidgeRegression import OnlineRidge
from src.OnlineRegressors.ConcreteRegressors.OnlineRidgeFSSCB import RidgeFSSCB
from src.OnlineRegressors.ConcreteRegressors.AdaptiveRegression import AdaptiveRegressor
from src.OnlineRegressors.ConcreteRegressors.SGDWrapper import SGDWrapper
matplotlib.use("tkagg")
from tqdm import tqdm


def test_adaptive(Xt, reg: Regressor, d, density, noise, n, eps=0.2, k=1, w_star=None):
    if w_star == None:
        w_star = sp.random(d, 1, density=density)
        w_star /= sp.linalg.norm(w_star, 1)

    reals = np.zeros(n)
    start = time.time()
    for i in tqdm(range(n)):
        xt = Xt[i].reshape(-1, 1)
        y_pred = reg.predict(xt)
        y_real = (w_star.T @ xt)[0][0] + np.random.normal(0, noise)
        reals[i] = w_star.T @ xt
        reals[i] = y_real
        reg.update(xt, y_pred, y_real)
    runtime = time.time() - start

    mse = np.mean((reg.pred_history - reals) ** 2)
    print()
    print(f"Final MSE: {mse}")
    # print(f"Weight Error: {np.linalg.norm(reg.w - w_star)}")

    # plt.plot(np.abs(reg.pred_history - reg.real_history))
    # plt.show()
    # MSE over time
    cumulative_mse = np.cumsum((reg.pred_history - reals) ** 2) / np.arange(1, len(reg.pred_history) + 1)
    return cumulative_mse, runtime, reals

def satisfies_RIP(X, w, eps, k):
    left = (1 - eps) * sp.linalg.norm(w)
    right = (1 + eps) * sp.linalg.norm(w)
    mid = (1 / np.sqrt(X.shape[0])) * np.linalg.norm(X @ w)
    return left <= mid <= right


d = 1000
density = 0.4
noise = 0.05
eps = 0.2
k = int(density * d)
n = int((1 / (eps * eps)) * k * np.log(eps * d / k))
n *= 2
n = 100
print(n)

w_star = sp.random(d, 1, density=density)
w_star /= sp.linalg.norm(w_star, 1)

Xt = np.random.normal(0, 1, size=(n, d))
print("Xt satisfies RIP: ", satisfies_RIP(Xt, w_star, 1/5, 3*k))
t0 = k * np.log(d) * np.log(n)
print("t0 is ", t0)
t0 = k
params = {"sigma": noise,
          "k": k,
          "k0": int(0.8 * d),
          "t0": 1
    }

# reg1 = AdaptiveRegressor(d, params)
params_r = {"lambda_reg": 0.2}
reg2 = RidgeFSSCB(d, params_r)

# res1, t1, reals1 = test_adaptive(Xt, reg1, d, density, noise, n, w_star=w_star)
res2, t2, reals2 = test_adaptive(Xt, reg2, d, density, noise, n, w_star=w_star)

# print(f"Adaptive runtime: {t1}")
print(f"Ridge runtime: {t2}")


# plt.plot(res1, label="POSLR")
plt.plot(res2, label="Ridge")
plt.legend()
plt.xlabel("Rounds")
plt.ylabel("MSE")
plt.title("Average MSE")
plt.show()

# print("W star is ", w_star.todense())
# print("POSLR w is ", reg1.w)
# plt.scatter(reg1.x_history, reals1, label="Real data")
# plt.scatter(reg1.x_history, reg1.pred_history, label="Predicted")
# plt.legend()
# plt.title("Real vs Adaptive")
# plt.show()


# plt.plot(reg1.regret(w_star, reals=reals1), label="POSLR")
plt.plot(reg2.regret(w_star), label="Ridge")
plt.legend()
plt.xlabel("Rounds")
plt.ylabel("Regret")
plt.title("Average Regret")
plt.show()



# plt.scatter(reals1, reg1.pred_history, label="POSLR", alpha=0.4)
plt.scatter(reals2, reg2.pred_history, label="Ridge", alpha=0.4)
# x = np.linspace(np.min(reals1), np.max(reals1), 100)
# plt.plot(x, x)
plt.legend()
plt.title("Predictions")
plt.show()

