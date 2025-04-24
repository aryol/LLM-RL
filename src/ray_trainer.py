from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import collate_fn
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from omegaconf import OmegaConf, open_dict
import wandb
import numpy as np

from src.utils.curriculum_dataset_wrapper import RatioAttemptsVariablesActor, \
                                                    CurriculumDatasetWrapper, PerSampleCurriculumDatasetWrapper

class RayPPOTrainerNonParquetteDataset(RayPPOTrainer):
    """
    This class is used to train a PPO model with a non-parquet dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size

        # taking care of the cases where a chat template is not defined. 
        if self.tokenizer.chat_template is None:
            print("No chat template found. Setting a custom one. it should take care of add_generation_prompt")
            self.tokenizer.chat_template = """{% for message in messages -%}
                                            {{ message['role'] }}: {{ message['content'] }}
                                            {% endfor -%}{% if add_generation_prompt -%}
                                            assistant: {% endif %}"""

        print(self.config.data.train_files)
        self.train_dataset = AdaptiveRLHFDataset(type=self.config.data.train_dataset_type, curriculum_config=self.config.data.curriculum_config,
                                        parquet_files=self.config.data.train_files,
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

        self.val_dataset = AdaptiveRLHFDataset(type='base', curriculum_config=self.config.data.curriculum_config,
                                    parquet_files=self.config.data.val_files,
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
        self.curriculum_config = kwargs.pop('curriculum_config', {})
        print(args, kwargs)
        super().__init__(*args, **kwargs)

    def _read_files_and_tokenize(self):
        super()._read_files_and_tokenize()
        # pass to curriculum dataset wrapper
        ratio_actor = RatioAttemptsVariablesActor.remote(
            dataset_length=len(self.dataframe),
            min_ratio=0.0,
            max_ratio=0.9,
            moving_avg_alpha=0.8,
            reward_threshold=0.5
        )
        if self.type == 'base':
            self.dataframe = CurriculumDatasetWrapper(self.dataframe, ratio_attempts_var_actor=ratio_actor, **self.curriculum_config)
        elif self.type == 'adaptive':
            self.dataframe = PerSampleCurriculumDatasetWrapper(self.dataframe, ratio_attempts_var_actor=ratio_actor, **self.curriculum_config)
        else:
            raise NotImplementedError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        chat = row_dict.pop(self.prompt_key)

        # add the partial answer to the chat, if provided
        if row_dict['extra_info'].get('partial_answer', None) is not None:
            partial_answer = row_dict['extra_info']['partial_answer']
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
    
