import os
import random
import json 

class GSM8KReward:
    def __init__(self, **kwargs):
        self.LOG_FILE = kwargs.get("LOG_FILE", "completion_logs.json")
        self.prob_save = kwargs.get("prob_save", 0.05)
        self.is_main_process = False

    def CorrectnessReward(self, completions, prompts, target, **kwargs):
        """
        Evaluates completions based on the correctness of the final answer. 

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
        
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        log_entries = []  # Store log data
        for completion, prompt, gt in zip(completions, prompts, target):
            try:
                if "####" in completion:
                    answer = int(completion.split("####")[1].strip())
                    if answer == int(gt):
                        reward = 1.0
                    else:
                        reward = 0.0
                else:
                    reward = 0.0
            except Exception:
                reward = 0.0
            rewards.append(reward)
            # Log this completion
            log_entries.append({
                "prompt": prompt,
                "completion": completion,
                "target": gt,
                "reward": reward
            })
        # Save logs
        if self.is_main_process and random.random() < self.prob_save:
            if not os.path.exists(self.LOG_FILE):
                with open(self.LOG_FILE, "w") as f:
                    json.dump([], f)  # Initialize empty list

            with open(self.LOG_FILE, "r+") as f:
                logs = json.load(f)
                logs.extend(log_entries)  # Append new logs
                f.seek(0)
                json.dump(logs, f, indent=4)

        return rewards

def ExtractAnswerFromDataset(text):
    """
    Extracts the answer from the dataset.
    The dataset separates the answer using the '####' delimiter.
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def FormatRewardFunction(completions, target, **kwargs):
    """
    Format: thinking process \n #### answer
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion in completions:
      try:
        if "####" in completion:
            answer = int(completion.split("####")[1].strip())
            rewards.append(1.0)
        else:
            rewards.append(0.0)
      except Exception:
        rewards.append(0.0)
    return rewards