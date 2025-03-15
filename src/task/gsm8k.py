import os
import random

def GSM8KReward(completions, prompts, target, **kwargs):
    """
    Evaluates completions based on the correctness of the final answer. 

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, prompt, gt in zip(completions, prompts, target):
      try:
        if random.random() < 1:  # 1% chance to write samples into a file
          os.makedirs("completion_samples", exist_ok=True)
          log_file = os.path.join("completion_samples", "completion_samples_answer.txt")
          with open(log_file, "a") as f:
            f.write(f"\n\n==============\n")
            f.write(prompt)
            f.write("\n$$\n")
            f.write(completion)

        if "####" in completion:
            answer = int(completion.split("####")[1].strip())
            if answer == int(gt):
                rewards.append(1.0)
                if random.random() < 1:  # 10% chance to write fully successful samples into a file
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "success_completion_samples_answer.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(prompt)
                        f.write("\n$$\n")
                        f.write(completion)
            else:
               rewards.append(0.0)
        else:
            rewards.append(0.0)
      except Exception:
        rewards.append(0.0)
    return rewards