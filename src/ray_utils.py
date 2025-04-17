from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import collate_fn
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from omegaconf import OmegaConf, open_dict
import os


class RayPPOTrainerNonParquetteDataset(RayPPOTrainer):
    """
    This class is used to train a PPO model with a non-parquet dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        print(self.config.data.train_files)
        self.train_dataset = AdaptiveRLHFDataset(type=self.config.data.train_dataset_type, parquet_files=self.config.data.train_files,
                                        tokenizer=self.tokenizer,
                                        processor=self.processor,
                                        prompt_key=self.config.data.prompt_key,
                                        image_key=self.config.data.get('image_key', 'images'),
                                        max_prompt_length=self.config.data.max_prompt_length,
                                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        truncation=self.config.data.get('truncation', 'error'),
                                        filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                                        num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                batch_size=self.config.data.get('gen_batch_size',
                                                                                self.config.data.train_batch_size),
                                                num_workers=8,
                                                drop_last=True,
                                                collate_fn=collate_fn,
                                                sampler=sampler)

        self.val_dataset = AdaptiveRLHFDataset(type='base', parquet_files=self.config.data.val_files,
                                    tokenizer=self.tokenizer,
                                    processor=self.processor,
                                    prompt_key=self.config.data.prompt_key,
                                    image_key=self.config.data.get('image_key', 'images'),
                                    max_prompt_length=self.config.data.max_prompt_length,
                                    return_raw_chat=self.config.data.get('return_raw_chat', False),
                                    truncation=self.config.data.get('truncation', 'error'),
                                    filter_overlong_prompts=self.config.data.filter_overlong_prompts,
                                    num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps
        
    def log(self, data, reward_extra_info):
        portions = reward_extra_info['portions']
        mean_portion = np.mean(portions)
        std_portion = np.std(portions)
        is_train_set =  data.non_tensor_batch['extra_info'][0]['split']
        stage = 'train' if is_train_set=='train' else 'val' 
        wandb.log({f'portions/{stage}_portions_mean': mean_portion}, step=self.global_steps)
        wandb.log({f'portions/{stage}_portions_std': std_portion}, step=self.global_steps)

    def update_datasets_with_ratios(self, data, scores, reward_extra_info):
        ids = reward_extra_info['index']
        portions = reward_extra_info['portions']
        if data.non_tensor_batch['extra_info'][0]['split'] == 'train':
            # Update the training dataset
            self.train_dataset.dataframe.ratio_actor.update_attempted_ratios.remote([(ids, portions, scores)])
            self.train_dataset.dataframe.ratio_actor.set_global_step.remote(self.global_steps)
            self.train_dataset.dataframe.sync_with_all_datasets()


from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

class AdaptiveRLHFDataset(RLHFDataset):
    """
    This class is used to train a PPO model with a non-parquet dataset.
    """

    def __init__(self, *args, **kwargs):
        self.type = kwargs.pop('type', 'base')
        print(args, kwargs)
        super().__init__(*args, **kwargs)

    def _read_files_and_tokenize(self):
        super()._read_files_and_tokenize()
        # pass to curriculum dataset wrapper
        ratio_actor = RatioAttemptsVarActor.remote(
            dataset_length=len(self.dataframe),
            min_ratio=0.0,
            max_ratio=0.9,
            moving_avg_alpha=0.8,
            reward_threshold=0.5
        )
        if self.type == 'base':
            self.dataframe = CurriculumDatasetWrapper(self.dataframe, ratio_actor)
        elif self.type == 'adaptive':
            self.dataframe = PerSampleCurriculumDatasetWrapper(self.dataframe, ratio_actor)
        else:
            raise NotImplementedError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        chat = row_dict.pop(self.prompt_key)

        if row_dict['extra_info'].get('partial_answer', None) is not None:
            partial_answer = row_dict['extra_info'].pop('partial_answer')
            chat.append({'role': 'assistant', 'content': partial_answer})
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, continue_final_message=True, tokenize=False)
        else: 
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        
        raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict 
    
import ray
import numpy as np

@ray.remote
class RatioAttemptsVarActor:
    def __init__(self, dataset_length, min_ratio=0.0, max_ratio=0.9, moving_avg_alpha=0.8, reward_threshold=0.5):
        self.attempted_ratios_list = [[] for _ in range(dataset_length)]
        self.newly_added_ids = set()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.reward_threshold = reward_threshold
        self.moving_avg_alpha = moving_avg_alpha
        self.global_step = 0
        self.mean_min_ratio = min_ratio
        self.mean_max_ratio = max_ratio        
        self.max_per_sample_ratio = np.ones(dataset_length) * self.max_ratio

    def update_attempted_ratios(self, gathered_data):
        for ids, portions, rewards in gathered_data:
            for id_, portion, reward in zip(ids, portions, rewards):
                entry_list = self.attempted_ratios_list[id_]
                if not entry_list or entry_list[-1]['portion'] != portion:
                    entry_list.append({'portion': portion, 'reward': [reward]})
                else:
                    entry_list[-1]['reward'].append(reward)
                self.newly_added_ids.add(id_)
        return 1

    def update_min_max_avg_ratios(self,):
        # Update global mean bounds using newly added IDs
        avg_macro_upper = 0
        avg_macro_lower = 0
        for id_ in self.newly_added_ids:
            entry = self.attempted_ratios_list[id_][-1]
            avg_reward = np.mean(entry['reward'])
            if avg_reward < self.reward_threshold:
                avg_macro_lower += entry['portion']
                avg_macro_upper += self.max_per_sample_ratio[id_]
            else:
                avg_macro_upper += entry['portion']
                avg_macro_lower += self.min_ratio
                self.max_per_sample_ratio[id_] = entry['portion']

        count = len(self.newly_added_ids)
        if count > 0:
            self.mean_max_ratio = (
                self.moving_avg_alpha * self.mean_max_ratio +
                (1 - self.moving_avg_alpha) * (avg_macro_upper / count)
            )
            self.mean_min_ratio = (
                self.moving_avg_alpha * self.mean_min_ratio +
                (1 - self.moving_avg_alpha) * (avg_macro_lower / count)
            )

        self.newly_added_ids.clear()
        return 1

    def set_global_step(self, step):
        self.global_step = step
        return 1

    def get_newly_added_ids(self):
        return list(self.newly_added_ids)

    def get_max_per_sample_ratio(self):
        return self.max_per_sample_ratio
    
    def get_sample_attempted_ratio_list(self):
        return self.attempted_ratios_list
    
    def get_min_mean_ratio(self):
        return self.mean_min_ratio

    def get_max_mean_ratio(self):
        return self.mean_max_ratio
    
    def get_global_step(self):
        return self.global_step

    def get_state_summary(self):
        return {
            'mean_min_ratio': self.mean_min_ratio,
            'mean_max_ratio': self.mean_max_ratio,
            'num_samples_with_history': sum(1 for x in self.attempted_ratios_list if x),
            'num_recently_updated': len(self.newly_added_ids)
        }

class CurriculumDatasetWrapper:
    def __init__(self, dataset, ratio_attempts_var_actor, initial_portion=0.0, prompt_key='prompt', target_key='answer', ):
        self.dataset = dataset  # Keep as HF Dataset
        self.ground_truth_portion_dist = initial_portion  # Start with a small proportion
        self.prompt_key = prompt_key
        self.target_key = target_key
        self.ratio_actor = ratio_attempts_var_actor
        self.global_step = 0
        self.portions = []

    def __getitem__(self, idx):
        """Retrieves a dataset sample with a dynamically adjusted reasoning portion."""
        sample = self.dataset[idx]  # Directly access HF dataset  
        portion = self._compute_portion_for_sample(idx)
        # self.global_step = ray.get(self.ratio_actor.get_global_step.remote())
        return self._apply_portion_to_sample_(sample, portion)

    def _compute_portion_for_sample(self, idx):
        portion = self.sample_portion(seed=self.return_seed(idx), size=1)[0] # Get current difficulty proportion
        return portion
    
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

    def _apply_portion_to_sample_(self, sample, portion):
        # Modify reasoning exposure based on `portion`
        if portion > 0.0:
            reasoning_steps = sample['extra_info'].get(self.target_key, '')
            word_list = reasoning_steps.split(' ')
            cut_answer = ' '.join(word_list[:int(len(word_list) * portion)])  # Partial CoT answer
            completion = ''.join(word_list[int(len(word_list) * portion):])  # Remaining CoT answer
            sample['extra_info']['partial_answer'] = cut_answer
            sample['extra_info']['completion'] = completion
            sample['extra_info']['portion'] = portion
        return sample

    def return_seed(self, idx):
        return self.global_step + 1024 * idx

    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        """Iterates over the dataset."""
        for i in range(len(self.dataset)):
            yield self[i]


import numpy as np
class PerSampleCurriculumDatasetWrapper(CurriculumDatasetWrapper):
    def __init__(self, dataset, ratio_attempts_var_actor, initial_portion=0.0, prompt_key='prompt', target_key='answer',
                 reward_threshold=0.5):
        super().__init__(dataset, ratio_attempts_var_actor, initial_portion, prompt_key, target_key)
        self.reward_threshold = reward_threshold
        self.min_mean_ratio = 0.0
        self.max_mean_ratio = 0.9
        self.max_per_sample_ratio = np.ones(len(self.dataset)) * self.max_mean_ratio
        self.attempted_ratio_list = [[] for _ in range(len(self.dataset))]  # Store attempted ratios for each sample
        self.global_step = 0

    def _compute_portion_for_sample(self, idx):
        upper_bound = self.max_per_sample_ratio[idx]
        lower_bound = 0.0 # because we never go back/ Ma be aghab bar nemigardim
        attempted_ratio_list = self.attempted_ratio_list[idx]

        if attempted_ratio_list:
            last_gen_portion = attempted_ratio_list[-1]['portion']
            last_gen_avg_reward = np.mean(attempted_ratio_list[-1]['reward'])
            if last_gen_avg_reward < self.reward_threshold:
                lower_bound = last_gen_portion
            else:
                upper_bound = last_gen_portion
        else:
            # moving average of the min and max ratio
            lower_bound = self.min_mean_ratio
            upper_bound = self.max_mean_ratio
        
        portion = self._sample_ratio_with_seed(seed=self.return_seed(idx), size=1, lower_bound=lower_bound, upper_bound=upper_bound)[0]
        return portion
    
    def sync_with_all_datasets(self):
        # Sync the ratio actor with all datasets
        ray.get(self.ratio_actor.update_min_max_avg_ratios.remote())
        # computing the lowerbound and upperbounds for portion sampling
        self.max_per_sample_ratio = ray.get(self.ratio_actor.get_max_per_sample_ratio.remote())
        self.min_mean_ratio = ray.get(self.ratio_actor.get_min_mean_ratio.remote()) # because we never go back/ Ma be aghab bar nemigardim
        self.attempted_ratio_list = ray.get(self.ratio_actor.get_sample_attempted_ratio_list.remote())
        self.global_step = ray.get(self.ratio_actor.get_global_step.remote())

    # Sample new portion from updated uniform distribution
    def _sample_ratio_with_seed(self, seed=42, size=1, lower_bound=0.0, upper_bound=1.0):
        return np.random.default_rng(seed).uniform(low=lower_bound, high=upper_bound, size=size)

    def set_portion(self, portion):
        raise NotImplementedError("Portion setting is not supported for per-sample curriculum learning.")


from verl.workers.reward_manager import NaiveRewardManager
from verl import DataProto
from collections import defaultdict
import wandb
class NaiveRewardManagerWithPortionLogging(NaiveRewardManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        reward_extra_info['portions'] = []
        reward_extra_info['index'] = data.non_tensor_batch['index']

        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            reward_extra_info['portions'].append(data_item.non_tensor_batch['extra_info'].get('portion', 0.0))
            scores.append(score)
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        self.trainer.log(data, reward_extra_info)
        self.trainer.update_datasets_with_ratios(data, scores ,reward_extra_info)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor