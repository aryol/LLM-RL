import os
import random
import json 
from .utils import BaseReward

class GSM8KReward(BaseReward):
    def __init__(self, **kwargs):
        super(GSM8KReward, self).__init__(**kwargs)

    def CorrectnessReward(self, completions, prompts, target, do_log = True, do_update=True, **kwargs):
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
        for completion, prompt, gt, idd in zip(completions, prompts, target, kwargs['id']):
            try:
                if "####" in completion:
                    answer = int(completion.split("####")[1].strip().replace(',', ''))
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
                "reward": reward,
                "id": idd
            })
        # Save logs
        if do_log:
            self.log_completions_to_file(log_entries)
        if do_update:
            self.update_datasets_with_ratios({'prompt': prompts, 'target': target, **kwargs}, rewards)
        return rewards


def ExtractAnswerFromDataset(text):
    """
    Extracts the answer from the dataset.
    The dataset separates the answer using the '####' delimiter.
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def FormatRewardFunction(completions, coef=1.0, **kwargs):
    """
    Format: thinking process \n #### answer
    Args:
        completions (list[str]): Generated outputs
      
      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion in completions:
      try:
        if "####" in completion:
            # answer = int(completion.split("####")[1].strip())
            rewards.append(coef)
        else:
            rewards.append(0.0)
      except Exception:
        rewards.append(0.0)
    return rewards