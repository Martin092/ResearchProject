from tqdm import tqdm

from Environments import LinearEnvironment
from Learners import ETCLearner
from Learners import FSSquareCB
from Learners import SquareCB
from src.Environments.LinearBandits.SquareCBEnvironment import SquareCBEnvironment
from utility.Logger import ResultLogger
from utility.SettingsSimulator import SettingsSimulator
import matplotlib
matplotlib.use("tkagg")
import numpy as np
import matplotlib.pyplot as plt

def main():

    # simple_example()
    complex_simulation()


def complex_simulation():

    settings_dir = "../configurations/presentation_plots"
    simulator = SettingsSimulator(settings_dir, "fsscb_informed.json")

    simulator.simulate_all()

    # fsscb_tune()

    # ks = np.arange(5, 101, 5)
    # plot_regret_k_fsscb_random(ks, 15)
    # plot_regret_k_fsscb_bayes(ks, 15)
    # plot_regret_k_squarecb_bayesian(ks, 15)
    # plot_regret_k_squarecb_fsslr(ks, 15)
    # plt.title("Final regret with respect to size of feature subset")
    # plt.xlabel("Number of features")
    # plt.ylabel("Final regret")
    # plt.legend()
    # plt.savefig("varying_k_regret")
    # plt.show()

def plot_regret_k_fsscb_random(ks, n=5):
    logger = ResultLogger("FS-SCB random")
    logger.new_log()
    regrets = np.zeros(len(ks))
    for _ in range(n):
        for i, k in tqdm(enumerate(ks)):
            env = SquareCBEnvironment({
                "d": 100,
                "sparsity": 0.05,
                "noise": 0.01,
                "k": 100
            })

            params = {
                "l_rate": 0.5,
                "strategy": "random",
                "num_features": k,
                "learn_rate": 800,
                "strat_params": {
                    "gibbs_iterations": 1000,
                    "M": 10,
                    "s": 5,
                    "warmup": 100
                }
            }

            learner = FSSquareCB(800, params)

            learner.run(env, logger)
            regrets[i] += env.get_cumulative_regret()

    plt.plot(ks, regrets/n, label="FS-SCB random")

def plot_regret_k_fsscb_bayes(ks, n=5):
    logger = ResultLogger("FS-SCB bayes")
    logger.new_log()
    regrets = np.zeros(len(ks))
    for _ in range(n):
        for i, k in tqdm(enumerate(ks)):
            env = SquareCBEnvironment({
                "d": 100,
                "sparsity": 0.05,
                "noise": 0.01,
                "k": 100
            })

            params = {
                "l_rate": 0.5,
                "strategy": "bayesian",
                "num_features": k,
                "learn_rate": 800,
                "strat_params": {
                    "gibbs_iterations": 1000,
                    "M": 10,
                    "s": 5,
                    "warmup": 100
                }
            }

            learner = FSSquareCB(800, params)

            learner.run(env, logger)
            regrets[i] += env.get_cumulative_regret()

    plt.plot(ks, regrets/n, label="FS-SCB Bayesian")

def plot_regret_k_squarecb_fsslr(ks, n=5):
    logger = ResultLogger("SquareCB with FSSLR")
    logger.new_log()
    regrets = np.zeros(len(ks))
    for _ in range(n):
        for i, k in tqdm(enumerate(ks)):
            env = SquareCBEnvironment({
                "d": 100,
                "sparsity": 0.05,
                "noise": 0.01,
                "k": 100
            })

            params = {
                "learn_rate": 200,
                "reg": "AdaptiveRegressor",
                "params_reg": {
                  "sigma": 0.01,
                  "k": 5,
                  "k0": k,
                  "t0": 1,
                  "C": 0.1,
                  "delta": 0.95
                }
            }

            learner = SquareCB(800, params)

            learner.run(env, logger)
            regrets[i] = env.get_cumulative_regret()

    plt.plot(ks, regrets/n, label="SquareCB + FSSLR")

def plot_regret_k_squarecb_bayesian(ks, n=5):
    logger = ResultLogger("SquareCB Bayesian")
    logger.new_log()
    regrets = np.zeros(len(ks))
    for _ in range(n):
        for i, k in tqdm(enumerate(ks)):
            env = SquareCBEnvironment({
                "d": 100,
                "sparsity": 0.05,
                "noise": 0.01,
                "k": 100
            })

            params = {
                "learn_rate": 200,
                "reg": "BayesianSelection",
                "params_reg": {
                  "lambda_reg": 0.6,
                  "k": 40,
                  "full_x": False
                }
            }

            learner = SquareCB(800, params)

            learner.run(env, logger)
            regrets[i] = env.get_cumulative_regret()

    plt.plot(ks, regrets/n, label="SquareCB Bayesian")


def fsscb_tune():
    print("Beginning FS-SCB fine tuning")


    l_rates = [0.1, 0.5, 1, 2]
    learn_rate = [500, 700, 800, 1000]

    logger = ResultLogger("Fine Tuning of FS-SCB")
    logger.new_log()

    best_regret = float('inf')
    best_l = 0
    best_l2 = 0
    for l in l_rates:
        for l2 in learn_rate:

            cum_regret = 0
            for i in tqdm(range(5), f"l_rate={l}, learn_rate={l2}"):
                env = SquareCBEnvironment({
                    "d": 100,
                    "sparsity": 0.05,
                    "noise": 0.01,
                    "k": 100
                })

                params = {
                    "l_rate": l,
                    "strategy": "bayesian",
                    "num_features": 40,
                    "learn_rate": l2,
                    "strat_params": {
                        "gibbs_iterations": 700,
                        "M": 5,
                        "s": 5,
                        "warmup": 100
                    }
                }

                learner = FSSquareCB(500, params)


                learner.run(env, logger)
                cum_regret += env.get_cumulative_regret()

            cum_regret /= 5
            print(f"l_rate={l}, learn_rate={l2}; regret={cum_regret}")
            if cum_regret < best_regret:
                best_l = l
                best_l2 = l2
                best_regret = cum_regret

    print()
    print(f"BEST l_rate={best_l}, learn_rate={best_l2}; regret={best_regret}")
    print("Terminating the Bandit Simulation")


def simple_example():
    print("Beginning a Bandit Simulation")

    simulation_name = "Example_Simulation"

    env = LinearEnvironment({
        "d" : 2,
        "action_set" : list([ [x, y] for x in range(3) for y in range(4)]),
        "true_theta" : [5, -3],
        "sigma" : 1,
        "k" : 12
    })

    learner = ETCLearner(24, {})

    logger = ResultLogger(simulation_name)
    logger.new_log()

    learner.run(env, logger)
    rewards = learner.cum_reward()
    cum_regret = env.get_cumulative_regret()

    print("Terminating the Bandit Simulation")

    plt.title("Rewards over time")
    plt.plot(np.cumsum(rewards), label="ETC")
    plt.legend()
    plt.show()

    plt.title("Regret")
    plt.plot(cum_regret)
    plt.show()


if __name__ == "__main__":
    main()
