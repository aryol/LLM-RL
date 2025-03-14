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
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser

import code

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
    curriculum_strategy: str = "beta"


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
    return rewards

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    supports_system_prompt = True
    if model_args.model_name_or_path is not None and "mathstral" in model_args.model_name_or_path.lower():
       supports_system_prompt = False


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset from Hugging Face Hub
    train_dataset = load_dataset(script_args.dataset_id_or_path, 'main')['train']
    test_dataset = load_dataset(script_args.dataset_id_or_path, 'main')['test']
    # select a random subset of 50k samples
    train_dataset = train_dataset.shuffle(seed=42).select(range(7000))
    test_dataset = test_dataset.shuffle(seed=42).select(range(560))
    
    def generate_prompt(example):
        partial_answer = example.get("partial_answer", "")
        if supports_system_prompt:
            prefix = [{
                "role": "system",
                "content": DEFAULT_PROMPT
            },
            { 
                "role": "user",
                "content": "Here is the question:\n" + example['question']
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n" + partial_answer
            }]
        else:
            prefix = [{ 
                "role": "user",
                "content": DEFAULT_PROMPT + "\n#\n" "Here is the question you need to solve:\n" + example['question']
            },
            {
                "role": "assistant",
                "content": "Let me solve this step by step.\n" + partial_answer
            }]
        return {"prompt": tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True), "target": extract_answer_from_dataset(example["answer"])}


    # Wrap dataset with curriculum learning
    train_dataset = CurriculumDatasetWrapper(train_dataset, generate_prompt, initial_portion=0.1)
    # test_dataset = CurriculumDatasetWrapper(test_dataset, initial_portion=0.1)
     # convert our dataset to the r1 prompts
    test_dataset = test_dataset.map(lambda x: generate_prompt(x))

    # Choose curriculum strategy (Beta or Linear)
    if script_args.curriculum_strategy == "beta":
        curriculum_updater = EnvironmentPortionBetaUpdate( train_dataset,
            init_alpha=2, init_beta=5, final_alpha=5, final_beta=2, warmup_timesteps=5000, total_timesteps=50000
        )
    else:
        curriculum_updater = EnvironmentPortionLinearUpdate( train_dataset,
            lower_bound_init=0.1, lower_bound_final=0.9, upper_bound_init=0.3, upper_bound_final=1.0,
            warmup_timesteps=5000, total_timesteps=50000
        )
    
    # GRPO Trainer
    trainer = CustomizedGRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[format_reward_func, answer_reward_func],
      processing_class=tokenizer,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      peft_config=get_peft_config(model_args),
    )
    # Register the updater
    trainer.add_callback(curriculum_updater)

    if trainer.accelerator.is_main_process:
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Training/evaluation parameters {training_args}")
        wandb.init(project="progressive-RL", name="train_without_warning")

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


class CurriculumDatasetWrapper:
    def __init__(self, dataset, generate_prompt, initial_portion=0.1):
        self.dataset = dataset  # Keep as HF Dataset
        self.generate_prompt = generate_prompt
        self.ground_truth_portion_dist = initial_portion  # Start with a small proportion

    def set_portion(self, portion):
        """Updates the portion of ground-truth CoT reasoning used in training."""
        self.ground_truth_portion_dist = portion

    def sample_portion(self):
        """Returns the current proportion of ground-truth CoT to use."""
        if callable(self.ground_truth_portion_dist):
            return self.ground_truth_portion_dist()
        elif isinstance(self.ground_truth_portion_dist, float):
            return self.ground_truth_portion_dist
        else:
            raise ValueError("ground_truth_portion_dist should be a float or a callable")

    def __getitem__(self, idx):
        """Retrieves a dataset sample with a dynamically adjusted reasoning portion."""
        sample = self.dataset[idx]  # Directly access HF dataset
        portion = self.sample_portion()  # Get current difficulty proportion

        # Modify reasoning exposure based on `portion`
        reasoning_steps = sample.get('answer', "")  # Assume dataset has 'answer'
        word_list = reasoning_steps.split(' ')
        cut_answer = ' '.join(word_list[:int(len(word_list) * portion)])  # Partial CoT answer

        # Append the partial answer to the question
        sample['question'] = sample['question'] + " " + cut_answer

        # Apply the chat prompt formatting
        return self.generate_prompt(sample)

    def __len__(self):
        return len(self.dataset)

    def map(self, function, **kwargs):
        """Allows dataset-wide transformation using Hugging Face's .map() method."""
        return CurriculumDatasetWrapper(self.dataset.map(function, **kwargs), self.generate_prompt, self.ground_truth_portion_dist)

    def shuffle(self, seed=None):
        """Allows shuffling while keeping the curriculum learning functionality."""
        return CurriculumDatasetWrapper(self.dataset.shuffle(seed=seed), self.generate_prompt, self.ground_truth_portion_dist)

    def select(self, indices):
        """Allows selecting a subset of data."""
        return CurriculumDatasetWrapper(self.dataset.select(indices), self.generate_prompt, self.ground_truth_portion_dist)

        
from transformers import TrainerCallback
from functools import partial
import numpy as np

class EnvironmentPortionBaseUpdate(TrainerCallback):
    """
    Update the portion of the environment actions that is used for training, linearly according to the current timestep.
    """
    
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset  # Reference to the curriculum dataset

    def update(self, state):
        raise NotImplementedError

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the end of every step to update difficulty dynamically."""
        self.update(state)

    # def on_train_begin(self, args, state, control, **kwargs):
    #     pass
        
    # def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
    #     pass

    # def on_evaluate(self, args, state, control, **kwargs):
    #     pass

    # def on_predict(self, args, state, control, **kwargs):
    #     pass

    # def on_log(self, args, state, control, logs=None, **kwargs):
    #     pass

    # def on_train_end(self, args, state, control, **kwargs):
    #     pass
   


class EnvironmentPortionBetaUpdate(EnvironmentPortionBaseUpdate):
    def __init__(self, dataset, init_alpha, init_beta, final_alpha, final_beta, warmup_timesteps, total_timesteps):
        super().__init__(dataset)
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.final_alpha = final_alpha
        self.final_beta = final_beta
        self.warmup_timesteps = warmup_timesteps
        self.total_timesteps = total_timesteps

    def update(self, state):
        """Updates dataset difficulty using a beta distribution based on training progress."""
        print("update")
        current_step = state.global_step  # Use HF Trainer’s step counter

        if current_step < self.warmup_timesteps:
            alpha, beta = self.init_alpha, self.init_beta
        elif current_step < self.total_timesteps:
            alpha = self.init_alpha + (self.final_alpha - self.init_alpha) * ((current_step - self.warmup_timesteps) / (self.total_timesteps - self.warmup_timesteps))
            beta = self.init_beta + (self.final_beta - self.init_beta) * ((current_step - self.warmup_timesteps) / (self.total_timesteps - self.warmup_timesteps))
        else:
            alpha, beta = self.final_alpha, self.final_beta

        # Sample new portion from updated beta distribution
        portion_dist = np.random.default_rng().beta(alpha, beta)

        # Update dataset portion dynamically
        self.dataset.set_portion(portion_dist)



class EnvironmentPortionLinearUpdate(EnvironmentPortionBaseUpdate):
    def __init__(self, dataset, lower_bound_init, lower_bound_final, upper_bound_init, upper_bound_final, warmup_timesteps, total_timesteps):
        super().__init__(dataset)
        self.lower_bound_init = lower_bound_init
        self.lower_bound_final = lower_bound_final
        self.upper_bound_init = upper_bound_init
        self.upper_bound_final = upper_bound_final
        self.warmup_timesteps = warmup_timesteps
        self.total_timesteps = total_timesteps

    def update(self, state):
        """Updates dataset difficulty using a linear progression based on training progress."""
        print("update")
        current_step = state.global_step  # Use HF Trainer’s step counter

        if current_step < self.warmup_timesteps:
            lower_bound, upper_bound = self.lower_bound_init, self.upper_bound_init
        elif current_step < self.total_timesteps:
            lower_bound = self.lower_bound_init + (self.lower_bound_final - self.lower_bound_init) * ((current_step - self.warmup_timesteps) / (self.total_timesteps - self.warmup_timesteps))
            upper_bound = self.upper_bound_init + (self.upper_bound_final - self.upper_bound_init) * ((current_step - self.warmup_timesteps) / (self.total_timesteps - self.warmup_timesteps))
        else:
            lower_bound, upper_bound = self.lower_bound_final, self.upper_bound_final

        # Sample new portion from updated uniform distribution
        portion_dist = np.random.default_rng().uniform(low=lower_bound, high=upper_bound)

        # Update dataset portion dynamically
        self.dataset.set_portion(portion_dist)



def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()