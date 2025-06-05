from tqdm import tqdm

from Environments import LinearEnvironment
from Learners import ETCLearner
from Learners import FSSquareCB
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

    settings_dir = "../configurations/paper_figures"
    simulator = SettingsSimulator(settings_dir, "full_comparison_k=80.json")

    simulator.simulate_all("Cumulative regret with k=80")

    # fsscb_tune()

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
