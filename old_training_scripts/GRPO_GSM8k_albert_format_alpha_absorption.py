# Code partly taken from https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/scripts/run_r1_grpo.py
import wandb
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional, Sized, Union
import os
import random
import re 
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
from torch.utils.data import Sampler
from math import ceil
import numpy as np

train_epochs = 3.5
decay_epochs = 2


class CustomizedRepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.
        seed (`Optional[int]`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count
    

class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [0, 0, 1, 1, 2, 2, 3, 3]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

    def __iter__(self):
        indexes = [
            idx
            for idx in range(self.num_samples)
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class CustomizedGRPOTrainer(GRPOTrainer):
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        first_key = next(iter(logs.keys()))
        if first_key.startswith("eval_"):
            first_key = "_".join(first_key.split("_")[:-1])
            metrics = {f"{first_key}_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super(GRPOTrainer, self).log(logs, start_time) # Assuming transformers versions is >= 4.47
        self._metrics.clear()

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        # return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)
        return RepeatSampler(self.train_dataset, self.num_generations)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        # return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)
        return RepeatSampler(eval_dataset, self.num_generations)
   


DEFAULT_PROMPT = """
1. Always present the final answer on the last line of your response in the format:
    #### <answer>
    Ensure that the answer is a single number.
2. End each sentence with a newline character ('\n').
3. Perform any calculations within a <<...>> block before outputing the result of this calculation.
"""

WARNING_PROMPT = """
1. Always present the final answer on the last line of your response in the format:
    #### <answer>
    Ensure that the answer is a single number.
2. End each sentence with a newline character ('\n').
3. Perform any calculations within a <<...>> block before outputing the result of this calculation.
        
# Example of a problem

A colony of bees can contain up to 80000 individuals. In winter they are more exposed to death, and if the winter is really cold the bees can begin to die slowly. If the colony starts to lose 1200 bees per day, how many days will pass until the number of bees in the colony reaches a fourth of its initial number?

# A bad solution

The initial number of bees is 80000. A fourth of this number is 80000 / 4 = <<80000/4=20000>>20000 bees. 
The colony will lose 1200 bees per day. To find out how many days it will take for the colony to reach a fourth of its initial number, we need to divide the initial number by the number of bees lost per day and add 1 (since we want to know how many days it will take to reach the target number, not how many days it will take to reach the number below the target). 
So, the number of days is (80000 - 20000) / 1200 + 1 = 60000 / 1200 + 1 = 50 + 1 = 51 days. 

#### 51

# Feedback

Your approach and calculations are correct up until the final step. The mistake is in adding an extra day to the result. The calculation (80000 - 20000) / 1200 already gives you the number of days it takes to reach exactly 20000 bees, which is a fourth of the initial number. Adding an extra day is not necessary because the division already accounts for the time it takes to reach that specific number. Therefore, the correct answer is 50 days. In the future, ensure that you understand what each step of your calculation represents to avoid such mistakes. Also, make sure to follow the suggested format for presenting the final answer.
"""



def extract_answer_from_dataset(text):
    """
    Extracts the answer from the dataset.
    The dataset separates the answer using the '####' delimiter.
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Custom dataclasses
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = 'openai/gsm8k'
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

# Helper functions
def format_reward_func(completions, target, **kwargs):
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
        if random.random() < 0.1:  # 1% chance to write samples into a file
          os.makedirs("completion_samples", exist_ok=True)
          log_file = os.path.join("completion_samples", "completion_samples.txt")
          with open(log_file, "a") as f:
            f.write(f"\n\n==============\n")
            f.write(completion)
        
        # # Check if the format is correct
        # regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

        # match = re.search(regex, completion, re.DOTALL) 
        # # if the format is not correct, reward is 0
        # if match is None or len(match.groups()) != 2:
        #     rewards.append(0.0)
        # else:
        #     rewards.append(1.0)
        if "####" in completion:
            answer = int(completion.split("####")[1].strip())
            rewards.append(1.0)
        else:
            rewards.append(0.0)
      except Exception:
        rewards.append(0.0)
    return rewards

def answer_reward_func(completions, target, **kwargs):
    """
    Evaluates completions based on the correctness of the final answer. 

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
    
    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt in zip(completions, target):
      try:
        if random.random() < 0.1:  # 1% chance to write samples into a file
          os.makedirs("completion_samples", exist_ok=True)
          log_file = os.path.join("completion_samples", "completion_samples.txt")
          with open(log_file, "a") as f:
            f.write(f"\n\n==============\n")
            f.write(completion)

        if "####" in completion:
            answer = int(completion.split("####")[1].strip())
            if answer == int(gt):
                rewards.append(1.0)
                if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "success_completion_samples.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(completion)
            else:
               rewards.append(0.0)
        else:
            rewards.append(0.0)
      except Exception:
        rewards.append(0.0)

    #   try:
    #     # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
    #     completion = "<think>" + completion
    #     # Check if the format is correct
    #     match = re.search(r"<answer>(.*?)<\/answer>", completion)
    #     if match is None:
    #         rewards.append(0.0)
    #         continue
    #     # Extract the "answer" part from the completion
    #     equation = match.group(1).strip()
    #     # Extract all numbers from the equation
    #     final_ans = int(equation)
    #     if final_ans == int(gt):
    #         rewards.append(1.0)
    #         if random.random() < 0.10:  # 10% chance to write fully successful samples into a file
    #             os.makedirs("completion_samples", exist_ok=True)
    #             log_file = os.path.join("completion_samples", "success_completion_samples.txt")
    #             with open(log_file, "a") as f:
    #                 f.write(f"\n\n==============\n")
    #                 f.write(completion)
    #     else:
    #         rewards.append(0.0)
    #   except Exception:
    #         # If evaluation fails, reward is 0
    #         rewards.append(0.0) 
    return rewards

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    train_epochs = 3.6
    decay_epochs = 2

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset from Hugging Face Hub
    train_dataset = load_dataset(script_args.dataset_id_or_path, 'main')['train']
    test_dataset = load_dataset(script_args.dataset_id_or_path, 'main')['test']
    # select a random subset of 50k samples
    train_epochs = int(ceil(train_epochs))
    train_dataset = train_dataset.shuffle(seed=42).select(range(7000))
    augmented_train_dataset = concatenate_datasets([train_dataset for _ in range(train_epochs)])

    test_dataset = test_dataset.shuffle(seed=42).select(range(560))

    def generate_prompt_without_warning(example):
        prefix = [{
            "role": "system",
            "content": DEFAULT_PROMPT
          },
          { 
            "role": "user",
            "content": "Here is the question:" + example['question']
          },
          {
            "role": "assistant",
            "content": "Let me solve this step by step.\n"
          }]
        return {"prompt": tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True), "target": extract_answer_from_dataset(example["answer"])}

    def generate_prompt_with_warning(example):
        prefix = [{
            "role": "system",
            "content": WARNING_PROMPT
          },
          { 
            "role": "user",
            "content": "Here is the question:" + example['question']
          },
          {
            "role": "assistant",
            "content": "Let me solve this step by step.\n"
          }]
        return {"prompt": tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True), "target": extract_answer_from_dataset(example["answer"])}


    rng_warning = np.random.default_rng(77)

    def generate_random_prompt(example, idx, rng):
        p = 1 - idx / (len(train_dataset) * decay_epochs)
        if p < 0.0: 
            p = 0.00
        use_warning = rng.choice([False, True], p=[1 - p, p])
        if use_warning:
            return generate_prompt_with_warning(example)
        else:
            return generate_prompt_without_warning(example)


    # convert our dataset to the r1 prompts
    train_dataset = augmented_train_dataset.map(lambda x, idx: generate_random_prompt(x, idx, rng_warning), with_indices=True)

    test_dataset_with_warning = test_dataset.map(lambda x: generate_prompt_with_warning(x))
    test_dataset_without_warning = test_dataset.map(lambda x: generate_prompt_without_warning(x))



    
    # GRPO Trainer
    print(f"Training set size is: {len(train_dataset)}.")
    trainer = CustomizedGRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[format_reward_func, answer_reward_func],
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset={'with_warning': test_dataset_with_warning, 'wo_warning': test_dataset_without_warning},
      peft_config=get_peft_config(model_args),
    )

    if trainer.accelerator.is_main_process:
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Training/evaluation parameters {training_args}")
        wandb.init(project="progressive-RL", name="train_absorption")


    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None and trainer.accelerator.is_main_process:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    if trainer.accelerator.is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
        )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if trainer.accelerator.is_main_process:
        logger.info("*** Training complete ***")

    # Save model
    if trainer.accelerator.is_main_process:
        logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        logger.info(f"Tokenizer saved to {training_args.output_dir}")
        logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()