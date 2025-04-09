import os
import random
import json
import copy
class BaseReward():

    def __init__(self, dataset, LOG_FILE="completion_logs.json", val_LOG_FILE="completion_logs_val.json", prob_save=0.05):
        self.LOG_FILE = LOG_FILE
        self.val_LOG_FILE = val_LOG_FILE
        self.prob_save = prob_save
        self.dataset = dataset
        self.dataset_clone = copy.deepcopy(dataset)
        self.is_main_process = False

    def is_train_dataset_or_eval_dataset(self, target, idd):
        for i in range(len(target)):
            if idd[i] >= len(self.dataset):
                return False
            if target[i] != self.dataset_clone[idd[i]]['target']:
                return False

        return True

    def CorrectnessReward(self, completions, prompts, target, **kwargs):
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
        some_targets = [log_entries[i]['target'] for i in range(len(log_entries))]
        some_ids = [log_entries[i]['id'] for i in range(len(log_entries))]
        if self.is_train_dataset_or_eval_dataset(some_targets, some_ids):
            # Use the training log file
            log_file = self.LOG_FILE
        else:
            # Use the validation log file
            log_file = self.val_LOG_FILE
        if self.is_main_process and random.random() < self.prob_save:
                if not os.path.exists(log_file):
                    with open(log_file, "w") as f:
                        json.dump([], f)  # Initialize empty list

                with open(log_file, "r+") as f:
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
        if self.is_train_dataset_or_eval_dataset(kwargs['target'], ids):
            # Use the training dataset
            self.dataset.update_attempted_ratios(ids, portions, rewards)
        else:
            # do not update the validation dataset
            pass