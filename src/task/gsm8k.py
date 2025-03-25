import os
import random
import json 
from .utils import BaseReward

class GSM8KReward(BaseReward):
    def __init__(self, **kwargs):
        super(GSM8KReward, self).__init__(**kwargs)

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
        self.log_completions_to_file(log_entries)
        self.update_datasets_with_ratios(kwargs, rewards)
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