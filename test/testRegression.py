import time

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from src.OnlineRegressors.AbstractRegressor import Regressor
from src.OnlineRegressors.ConcreteRegressors.BayesianSelection import BayesianSelection
from src.OnlineRegressors.ConcreteRegressors.LinearRegression import LinearRegression
from src.OnlineRegressors.ConcreteRegressors.RidgeRegression import OnlineRidge
from src.OnlineRegressors.ConcreteRegressors.OnlineRidgeFSSCB import RidgeFSSCB
from src.OnlineRegressors.ConcreteRegressors.AdaptiveRegression import AdaptiveRegressor
from src.OnlineRegressors.ConcreteRegressors.SGDWrapper import SGDWrapper
matplotlib.use("tkagg")
from tqdm import tqdm


def test_adaptive(Xt, reg: Regressor, d, density, noise, n, eps=0.2, k=1, w_star=None):
    if w_star == None:
        w_star = sp.random(1, d, density=density)
        w_star /= sp.linalg.norm(w_star, 1)
        w_star = w_star.todense().flatten()

    reals = np.zeros(n)
    start = time.time()
    for i in tqdm(range(n)):
        xt = Xt[i]
        y_pred = reg.predict(xt)
        y_real = (w_star @ xt)[0] + np.random.normal(0, noise)
        reals[i] = w_star @ xt
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
    mid = (1 / np.sqrt(X.shape[0])) * np.linalg.norm(X @ w.T)
    return left <= mid <= right


d = 100
density = 0.05
noise = 0.01
eps = 0.2
k = int(density * d)
n = int((1 / (eps * eps)) * k * np.log(eps * d / k))
n *= 2
n = 5000
print(n)

w_star = sp.random(1, d, density=density)
w_star /= sp.linalg.norm(w_star, 1)

Xt = np.random.normal(0, 1, size=(n, d))
print("Xt satisfies RIP: ", satisfies_RIP(Xt, w_star, 1/5, 3*k))
t0 = k * np.log(d) * np.log(n)
print("t0 is ", t0)
t0 = k
params = {
          "sigma": 0.01,
          "k": 5,
          "k0": 80,
          "t0": 1,
          "C": 0.1,
          "delta": 0.95,
          "smart_sample": True
        }

# reg1 = AdaptiveRegressor(d, params)
params_r = {"lambda_reg": 0.6, "k": 10}
reg1 = BayesianSelection(d, params_r)
reg2 = RidgeFSSCB(d, params_r)
params["smart_sample"] = False
reg3 = AdaptiveRegressor(d, params)

res1, t1, reals1 = test_adaptive(Xt, reg1, d, density, noise, n, w_star=w_star)
res2, t2, reals2 = test_adaptive(Xt, reg2, d, density, noise, n, w_star=w_star)
res3, t3, reals3 = test_adaptive(Xt, reg3, d, density, noise, n, w_star=w_star)

print(f"Adaptive runtime: {t1}")
print(f"Ridge runtime: {t2}")
print(f"Dumb adaptive runtime: {t3}")

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle(f"Comparison of FSSLR, Bayesian and Ridge Regression (d={d}, s={int(d * density)})", fontsize=16)

# Plot MSE
line1, = axs[0].plot(res1, label="Bayesian")
line2, = axs[0].plot(res2, label="Ridge")
line3, = axs[0].plot(res3, label="FSSLR")
axs[0].set_xlabel("Rounds")
axs[0].set_ylabel("MSE")
axs[0].set_title("Average MSE")

# Plot Regret
axs[1].plot(np.sqrt(reg1.regret(w_star, reals=reals1)), label="Bayesian")
axs[1].plot(np.sqrt(reg2.regret(w_star)), label="Ridge")
axs[1].plot(np.sqrt(reg3.regret(w_star)), label="FSSLR")
axs[1].set_xlabel("Rounds")
axs[1].set_ylabel("Regret")
axs[1].set_title("Cumulative Regret")


fig.legend([line1, line2, line3], [f"Bayesian with k={params_r['k']}", "Ridge", f"FSSLR with k={params['k0']}"], loc='lower center', ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()





# # plt.scatter(reals1, reg1.pred_history, label="POSLR", alpha=0.4)
# plt.scatter(reals2, reg2.pred_history, label="Ridge", alpha=0.4)
# # x = np.linspace(np.min(reals1), np.max(reals1), 100)
# # plt.plot(x, x)
# plt.legend()
# plt.title("Predictions")
# plt.show()

