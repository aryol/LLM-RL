import os
import random
import json

class BaseReward():

    def __init__(self, dataset, LOG_FILE="completion_logs.json", prob_save=0.05):
        self.LOG_FILE = LOG_FILE
        self.prob_save = prob_save
        self.dataset = dataset
        self.is_main_process = False

    def CorrectnessReward(self, completions, prompts, target, do_log=True, **kwargs):
        """
        Evaluates completions based on the correctness of the final answer. 

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
        
        Returns:
            list[float]: Reward scores
        """
        raise NotImplementedError
    
    def log_completions_to_file(self, log_entries):
        """
        Logs completions to a file.

        Args:
            log_entries (list[dict]): List of log entries
        """
        if self.is_main_process and random.random() < self.prob_save:
                if not os.path.exists(self.LOG_FILE):
                    with open(self.LOG_FILE, "w") as f:
                        json.dump([], f)  # Initialize empty list

                with open(self.LOG_FILE, "r+") as f:
                    logs = json.load(f)
                    logs.extend(log_entries)  # Append new logs
                    f.seek(0)
                    json.dump(logs, f, indent=4)
    
    def update_datasets_with_ratios(self, kwargs, rewards):
        """
        Updates the datasets with ratios and rewards.

        """
        ids = kwargs['id']
        portions = kwargs['portion']
        self.dataset.update_attempted_ratios(ids, portions, rewards)