from transformers.trainer_utils import get_last_checkpoint
import os
import random

# Helper functions

def get_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

class CurriculumDatasetWrapper:
    def __init__(self, dataset, generate_prompt, initial_portion=0.0, prompt_key='prompt', target_key='target'):
        self.dataset = dataset  # Keep as HF Dataset
        self.generate_prompt = generate_prompt
        self.ground_truth_portion_dist = initial_portion  # Start with a small proportion
        self.prompt_key = prompt_key
        self.target_key = target_key
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
        reasoning_steps = sample.get(self.target_key, '')
        word_list = reasoning_steps.split(' ')
        cut_answer = ' '.join(word_list[:int(len(word_list) * portion)])  # Partial CoT answer

        # Apply the chat prompt formatting
        return self.generate_prompt({**sample, 'partial_target':cut_answer})

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
