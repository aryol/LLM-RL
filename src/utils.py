from transformers.trainer_utils import get_last_checkpoint
import os
import numpy as np
from functools import partial
import hydra
from os import getpid
from torch.distributed import all_gather_object, get_rank, get_world_size
import torch
import datasets

def get_checkpoint(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def return_generate_prompt(config, tokenizer):
    DEFAULT_PROMPT = config.task.default_prompt if config.task.get("default_prompt") is not None else "Solve the following math problem:"
    extract_answer_from_dataset = hydra.utils.get_method(config.task.extract_answer_from_dataset)

    supports_system_prompt = True
    if config.model.model_name_or_path is not None and "mathstral" in config.model.model_name_or_path.lower():
        supports_system_prompt = False
    supports_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None

    def generate_prompt(example, prompt_key="prompt", target_key="target"):
        partial_answer = example.get("partial_target", "")
        if supports_system_prompt:
            # prefix = [
            #     {"role": "system", "content": DEFAULT_PROMPT},
            #     {"role": "user", "content": "Here is the question:\n" + example[prompt_key]},
            #     {"role": "assistant", "content": "Let me solve this step by step.\n" + partial_answer}
            # ]
            if partial_answer == "":
                prefix = [
                    # {"role": "system", "content": DEFAULT_PROMPT},
                    {"role": "user", "content": example[prompt_key] + "\n" + DEFAULT_PROMPT},
                ]
            else:
                prefix = [
                    # {"role": "system", "content": DEFAULT_PROMPT},
                    {"role": "user", "content": example[prompt_key] + "\n" + DEFAULT_PROMPT},
                    {"role": "assistant", "content": partial_answer}
                ]

        else:
            prefix = [
                {"role": "user", "content": DEFAULT_PROMPT + "\n#\nHere is the question you need to solve:\n" + example[prompt_key]},
                {"role": "assistant", "content": "Let me solve this step by step.\n" + partial_answer}
            ]

        if supports_chat_template:
            formatted_prompt = tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)
        else:
            # Manual fallback formatting for non-chat models like GPT2
            if supports_system_prompt:
                formatted_prompt = f"{DEFAULT_PROMPT}\n\nHere is the question:\n{example[prompt_key]}\n\nLet me solve this step by step.\n{partial_answer}"
            else:
                formatted_prompt = f"{DEFAULT_PROMPT}\n#\nHere is the question you need to solve:\n{example[prompt_key]}\n\nLet me solve this step by step.\n{partial_answer}"

        rest_of_keys = set(example.keys()) - set([prompt_key, target_key])
        rest_of_example = {k: example[k] for k in rest_of_keys}
        return {
            **rest_of_example,
            "prompt": formatted_prompt,
            "target": extract_answer_from_dataset(example[target_key])
        }

    return partial(generate_prompt, prompt_key=config.task.prompt_key, target_key=config.task.target_key)


class CurriculumDatasetWrapper:
    def __init__(self, dataset, generate_prompt, initial_portion=0.0, prompt_key='prompt', target_key='target',):
        self.dataset = dataset  # Keep as HF Dataset
        # self.attempted_ratios_list = [[]]*len(dataset) this will create a list of the same object! jesus christ
        self.attempted_ratios_list = [[] for _ in range(len(dataset))]
        self.generate_prompt = generate_prompt
        self.ground_truth_portion_dist = initial_portion  # Start with a small proportion
        self.prompt_key = prompt_key
        self.target_key = target_key
        self.newly_added_ids = set()
        self.flush_newly_added_ids = False
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
        completion = ''.join(word_list[int(len(word_list) * portion):])  # Remaining CoT answer
        self.flush_newly_added_ids = True
        # Apply the chat prompt formatting
        return self.generate_prompt({**sample, 'partial_target':cut_answer, 'completion':completion, 'portion':portion, 'id':idx})

    def return_seed(self, idx):
        return self.global_step + 1024 * idx

    def update_attempted_ratios(self, ids, portions, rewards):
        """Updates the dataset with the attempted ratios and rewards."""

        if self.flush_newly_added_ids:
            self.newly_added_ids.clear()
            self.flush_newly_added_ids = False

        if torch.distributed.is_initialized() and get_world_size() > 1:
            gathered_data = [None for _ in range(get_world_size())]
            all_gather_object(gathered_data, (ids, portions, rewards))
        else:
            gathered_data = [(ids, portions, rewards)]   

        for ids, portions, rewards in gathered_data:
            for i, (id_, portion, reward) in enumerate(zip(ids, portions, rewards)):
                if self.attempted_ratios_list[id_]==[]:
                    self.attempted_ratios_list[id_].append({'portion': portion, 'reward': [reward]})
                elif self.attempted_ratios_list[id_][-1]['portion'] != portion:
                    self.attempted_ratios_list[id_].append({'portion': portion, 'reward': [reward]})
                else:
                    self.attempted_ratios_list[id_][-1]['reward'].append(reward)
                self.newly_added_ids.add(id_)
        
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        """Iterates over the dataset."""
        for i in range(len(self.dataset)):
            yield self[i]

    # def map(self, function, **kwargs):
    #     """Allows dataset-wide transformation using Hugging Face's .map() method."""
    #     return CurriculumDatasetWrapper(self.dataset.map(function, **kwargs), self.generate_prompt, self.ground_truth_portion_dist)

    # def shuffle(self, seed=None):
    #     """Allows shuffling while keeping the curriculum learning functionality."""
    #     return CurriculumDatasetWrapper(self.dataset.shuffle(seed=seed), self.generate_prompt, self.ground_truth_portion_dist)

    # def select(self, indices):
    #     """Allows selecting a subset of data."""
    #     return CurriculumDatasetWrapper(self.dataset.select(indices), self.generate_prompt, self.ground_truth_portion_dist)


class PerSampleCurriculumDatasetWrapper(CurriculumDatasetWrapper):
    def __init__(self, dataset, generate_prompt, initial_portion=0.0, prompt_key='prompt', target_key='target', min_ratio=0.0, max_ratio=0.9, reward_threshold=0.5, moving_avg_alpha=0.8):
        super().__init__(dataset, generate_prompt, initial_portion, prompt_key, target_key)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.reward_threshold = reward_threshold
        self.moving_avg_alpha = moving_avg_alpha
        self.mean_min_ratio = min_ratio
        self.mean_max_ratio = max_ratio

        self.max_per_sample_ratio = np.ones(len(dataset)) * self.max_ratio

    def __getitem__(self, idx):
        """Retrieves a dataset sample with a dynamically adjusted reasoning portion."""
        sample = self.dataset[idx]  # Directly access HF dataset
        lower_bound = self.min_ratio
        upper_bound = self.max_per_sample_ratio[idx] # because we never go back/ Ma be aghab bar nemigardim
        if self.attempted_ratios_list[idx]:
            last_gen_portion = self.attempted_ratios_list[idx][-1]['portion']
            last_gen_avg_reward = np.mean(self.attempted_ratios_list[idx][-1]['reward'])
            if last_gen_avg_reward < self.reward_threshold:
                lower_bound = last_gen_portion
            else:
                upper_bound = last_gen_portion
        else:
            # moving average of the min and max ratio
            lower_bound = self.mean_min_ratio
            upper_bound = self.mean_max_ratio
                                  
        portion = self._sample_ratio_with_seed(seed=self.return_seed(idx), size=1, lower_bound=lower_bound, upper_bound=upper_bound)[0]
        self.portions.append(portion)
        self.flush_newly_added_ids = True
        # Modify reasoning exposure based on `portion`
        reasoning_steps = sample.get(self.target_key, '')
        word_list = reasoning_steps.split(' ')
        cut_answer = ' '.join(word_list[:int(len(word_list) * portion)])  # Partial CoT answer
        completion = ''.join(word_list[int(len(word_list) * portion):])
        # Apply the chat prompt formatting
        return self.generate_prompt({**sample, 'partial_target':cut_answer, 'completion':completion,
                                     'portion':portion, 'id':idx})

    def update_attempted_ratios(self, ids, portions, rewards):
        """Updates the dataset with the attempted ratios and rewards."""
        if self.flush_newly_added_ids and self.newly_added_ids:
            # compute moving averages of max and min ratios over samples 
            # (to be used in the first epoch when we havent seen any rewards for each sample yet.)
            avg_macro_batch_upperbound = 0
            avg_macro_batch_lowerbound = 0
            for id_ in self.newly_added_ids:
                last_gen_portion = self.attempted_ratios_list[id_][-1]['portion']
                last_gen_avg_reward = np.mean(self.attempted_ratios_list[id_][-1]['reward'])
                if last_gen_avg_reward < self.reward_threshold:
                    avg_macro_batch_lowerbound += last_gen_portion
                    avg_macro_batch_upperbound += self.max_per_sample_ratio[id_]
                else:
                    avg_macro_batch_upperbound += last_gen_portion
                    avg_macro_batch_lowerbound += self.min_ratio    
                    self.max_per_sample_ratio[id_] = last_gen_portion
        
            avg_macro_batch_upperbound /= len(self.newly_added_ids)
            avg_macro_batch_lowerbound /= len(self.newly_added_ids)

            self.mean_max_ratio = self.moving_avg_alpha * self.mean_max_ratio + (1 - self.moving_avg_alpha) * avg_macro_batch_upperbound
            self.mean_min_ratio = self.moving_avg_alpha * self.mean_min_ratio + (1 - self.moving_avg_alpha) * avg_macro_batch_lowerbound

        super().update_attempted_ratios(ids, portions, rewards)


    # Sample new portion from updated uniform distribution
    def _sample_ratio_with_seed(self, seed=42, size=1, lower_bound=0.0, upper_bound=1.0):
        return np.random.default_rng(seed).uniform(low=lower_bound, high=upper_bound, size=size)

    def set_portion(self, portion):
        raise NotImplementedError("Portion setting is not supported for per-sample curriculum learning.")