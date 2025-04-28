from Environments import LinearEnvironment
from Learners import ETCLearner

import itertools
import numpy as np
import matplotlib.pyplot as plt

def main():


    print("Beginning a Bandit Simulation")

    env = LinearEnvironment({
        "d" : 2,
        "action_set" : list([ [x, y] for x in range(3) for y in range(4)]),
        "true_theta" : [5, -3],
        "sigma" : 1,
        "k" : 12
    })

    learner = ETCLearner(24, {})

    learner.run(env)
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