from src.utility.Logger import ResultLogger
from src import Environments, Learners
from src.Environments import AbstractEnvironment
from src.Learners import AbstractLearner

import os.path
import json
from tqdm import trange



class SettingsSimulator:

    def __init__(self, settings_dir):

        self.settings_dir = settings_dir
        self.settings_path = os.path.join(settings_dir, "simulation_settings.json")

        self._read_settings()

        self.logger = ResultLogger(self.name)
        self.logger.new_log()


    def _read_settings(self):

        data = json.load(open(self.settings_path, mode = "r", encoding="utf-8"))

        assert data["simulations"] is not None
        assert data["name"] is not None

        self.name = data["name"]
        self.settings = data["simulations"]

        self.curr_simulation = 0
        self.num_simulations = len(self.settings)

        simulation_names = list(map(lambda sim : sim["name"], self.settings))

        # Make sure that simulation names are unique
        if len(simulation_names) != len(set(simulation_names)):
            raise RuntimeError("Simulation names are not unique")


    def simulate_next(self):

        if not self._has_next_simulation():
            return

        curr_settings = self.settings[self.curr_simulation]

        # Extract the parameters
        name = curr_settings["name"]
        trials = curr_settings["trials"]
        n = curr_settings["horizon"]

        env_cls = getattr(Environments, curr_settings["env"])
        learner_cls = getattr(Learners, curr_settings["learner"])


        for trial in trange(trials, desc=f"Running {name}"):
            # Set up the logger
            self.logger.set_simulation(name, trial + 1)

            # Instantiate a new copy of the environment and learner
            env : AbstractEnvironment = env_cls(curr_settings["env_config"])
            learner : AbstractLearner = learner_cls(n, curr_settings["learner_config"])

            learner.run(env, self.logger)

        self.curr_simulation += 1

    def simulate_all(self):

        while self._has_next_simulation():
            self.simulate_next()

    def _has_next_simulation(self):
        return self.curr_simulation < self.num_simulations