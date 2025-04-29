import os
import csv
from datetime import datetime
from typing import Optional


class ResultLogger:

    def __init__(self, name):


        self.base_dir = "../results"
        self.fields = ["Round", "Context", "Action", "Reward", "Cumulative Regret", "Time"]
        self.name = name
        self.log_dir: Optional[str] = None
        self.log_file_path: Optional[str] = None

    def new_log(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(self.base_dir, f"run_{self.name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=False)

        self.log_file_path = os.path.join(self.log_dir, "log.csv")

        with open(self.log_file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.fields)

    def log(self, t, context, action, reward, regret):

        row = [t, context, action, reward, regret, datetime.now().strftime("%Y/%m/%d-%H:%M:%S")]

        if not self.log_file_path:
            raise RuntimeError("Log file not initialized. Call create_log() first.")

        with open(self.log_file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)