import hydra   
from omegaconf import DictConfig, OmegaConf
from typing import Optional
from omegaconf import OmegaConf
from trl import GRPOConfig

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import os
import random
import wandb
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from trl import GRPOTrainer, get_peft_config, GRPOConfig
from transformers.trainer_utils import speed_metrics

import code

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

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


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def trainer(config):

    supports_system_prompt = True
    if config.model.model_name_or_path is not None and "mathstral" in config.model.model_name_or_path.lower():
       supports_system_prompt = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            config.model.tokenizer_name_or_path if config.model.get('tokenizer_name_or_path') else config.model.model_name_or_path
        ),
        revision=config.model.model_config.model_revision,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset from Hugging Face Hub
    train_dataset = load_dataset(config.task.dataset_id_or_path, 'main')['train']
    test_dataset = load_dataset(config.task.dataset_id_or_path, 'main')['test']
    # select a random subset of 50k samples
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)
    
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

    train_dataset = CurriculumDatasetWrapper(train_dataset, generate_prompt, initial_portion=0.0)
     # convert our dataset to the r1 prompts
    test_dataset = test_dataset.map(lambda x: generate_prompt(x))

    answer_reward_func = hydra.utils.get_method(config.task.reward_function,)

    training_args = OmegaConf.to_container(config.training_args)
    training_args = GRPOConfig(**training_args)
    model_args = hydra.utils.instantiate(config.model.model_config)
    # GRPO Trainer
    trainer = GRPOTrainer(
      model=config.model.model_name_or_path,
      reward_funcs=[format_reward_func, answer_reward_func],
      processing_class=tokenizer,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      peft_config=get_peft_config(model_args),
    )
    
    # Register the updater
    # Choose curriculum strategy (Beta or Linear, none)
    # Wrap dataset with curriculum learning
    if config.curriculum_strategy.get('_target_') is not None:
        curriculum_updater = hydra.utils.instantiate(config.curriculum_strategy, dataset=train_dataset)
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


####### utils code #######
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
          log_file = os.path.join("completion_samples", "completion_samples_format.txt")
          with open(log_file, "a") as f:
            f.write(f"\n\n==============\n")
            f.write(completion)

        if "####" in completion:
            answer = int(completion.split("####")[1].strip())
            rewards.append(1.0)
        else:
            rewards.append(0.0)
      except Exception:
        rewards.append(0.0)
    return rewards

def get_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

class CurriculumDatasetWrapper:
    def __init__(self, dataset, generate_prompt, initial_portion=0.0):
        self.dataset = dataset  # Keep as HF Dataset
        self.generate_prompt = generate_prompt
        self.ground_truth_portion_dist = initial_portion  # Start with a small proportion
        self.global_step = 0
        self.portions = []

    def set_portion(self, portion):
        """Updates the portion of ground-truth CoT reasoning used in training."""
        self.ground_truth_portion_dist = portion

    def sample_portion(self, seed=42, size=1):
        """Returns the current proportion of ground-truth CoT to use."""
        if callable(self.ground_truth_portion_dist):
            return self.ground_truth_portion_dist(seed, size)
        elif isinstance(self.ground_truth_portion_dist, float):
            return [self.ground_truth_portion_dist]
        else:
            raise ValueError("ground_truth_portion_dist should be a float or a callable")

    def __getitem__(self, idx):
        """Retrieves a dataset sample with a dynamically adjusted reasoning portion."""
        sample = self.dataset[idx]  # Directly access HF dataset
        portion = self.sample_portion(seed=self.return_seed(idx), size=1)[0] # Get current difficulty proportion
        self.portions.append(portion)
        # Modify reasoning exposure based on `portion`
        reasoning_steps = sample.get('answer', "")  # Assume dataset has 'answer'
        word_list = reasoning_steps.split(' ')
        cut_answer = ' '.join(word_list[:int(len(word_list) * portion)])  # Partial CoT answer

        # Append the partial answer to the question
        sample['question'] = sample['question']

        # Apply the chat prompt formatting
        return self.generate_prompt({**sample, 'partial_answer':cut_answer})

    def return_seed(self, idx):
        return self.global_step + 1024 * idx

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


@hydra.main(version_base="1.3", config_path="./config", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    outputs  = trainer(cfg)
    return outputs

if __name__ == "__main__":
    main()
