import hydra
from omegaconf import DictConfig, OmegaConf
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer
from typing import Optional
from datetime import datetime
import logging
import wandb
import rootutils
from transformers import TrainerCallback, AutoModelForCausalLM
from trl import DataCollatorForCompletionOnlyLM
from omegaconf import OmegaConf
import os

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

import numpy as np
from huggingface_hub import create_repo, HfApi
username = HfApi().whoami()["name"]

class PushToHubCallback(TrainerCallback):
    def __init__(self, tokenizer, base_name, dataset_name, prompt_name, push_dir="sft_checkpoints", push_to_hub=True):
        self.tokenizer = tokenizer
        self.base_name = base_name
        self.dataset_name = dataset_name
        self.prompt_name = prompt_name
        self.push_dir = push_dir
        self.trainer = None
        self.push_to_hub = push_to_hub

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return
        
        model_hub_name = f"sft-{self.base_name}-{self.dataset_name}-epoch{int(state.epoch)}-longsysprompt"
        repo_id = f"{username}/{model_hub_name}" 
        create_repo(repo_id, exist_ok=True)

        save_dir = os.path.join(self.trainer.args.output_dir, f"epoch-{int(state.epoch)}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.trainer.save_model(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        self.trainer.model.push_to_hub(repo_id)
        self.trainer.tokenizer.push_to_hub(repo_id)



def trainer(config):
    global tokenizer

    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))  # Or another port
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()
    
    model = AutoModelForCausalLM.from_pretrained(config.model.model_name_or_path,)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name_or_path if config.model.get("tokenizer_name_or_path") else config.model.model_name_or_path,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
        model.config.pad_token_id = tokenizer.pad_token_id # updating model config
        model.resize_token_embeddings(len(tokenizer) + 1)

    default_COT_prompt = config.task.default_prompt
    supports_system_prompt = hasattr(tokenizer, "apply_system_prompt") and tokenizer.system_prompt is not None
    supports_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
    
    def generate_prompt(example, prompt_key, target_key):
        partial_answer = example.get("partial_target", "")
        if supports_system_prompt:
            prefix = [
                {"role": "system", "content": default_COT_prompt},
                {"role": "user", "content": "Here is the question:\n" + example[prompt_key]},
                {"role": "assistant", "content": "Let me solve this step by step.\n" + partial_answer}
            ]
        else:
            prefix = [
                {"role": "user", "content": default_COT_prompt + "\n#\nHere is the question you need to solve:\n" + example[prompt_key]},
                {"role": "assistant", "content": "Let me solve this step by step.\n" + partial_answer}
            ]

        if supports_chat_template:
            formatted_prompt = tokenizer.apply_chat_template(prefix, tokenize=False, continue_final_message=True)
        else:
            # Manual fallback formatting for non-chat models like GPT2
            if supports_system_prompt:
                formatted_prompt = f"{default_COT_prompt}\n\nHere is the question:\n{example[prompt_key]}\n\nLet me solve this step by step.\n{partial_answer}"
            else:
                formatted_prompt = f"{default_COT_prompt}\n#\nHere is the question you need to solve:\n{example[prompt_key]}\n\nLet me solve this step by step.\n{partial_answer}"

        return {'prompt': formatted_prompt, 'completion': example[target_key]+tokenizer.eos_token}

    dataset = hydra.utils.instantiate(config.task.dataset, _convert_="all")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(20)) # for debugging
    val_dataset = dataset["test"].shuffle(seed=42).select(range(20)) # for debugging
    
    prompt_key = config.task.prompt_key
    target_key = config.task.target_key
    train_dataset = train_dataset.map(
        lambda x: generate_prompt(x, prompt_key, target_key),
        remove_columns=train_dataset.column_names,)
    val_dataset = val_dataset.map(
        lambda x: generate_prompt(x, prompt_key, target_key),
        remove_columns=val_dataset.column_names,)

    wandb.init(**OmegaConf.to_container(config.wandb_config, resolve=True))
    
    callbacks = [
        PushToHubCallback(
            tokenizer=tokenizer,
            base_name=config.model.model_name_or_path.split("/")[-1],
            dataset_name=config.task.task_name,
            prompt_name=config.task.default_prompt[:10], 
            push_dir=config.trainer_args.logging_dir,
        )
    ]

    training_args = hydra.utils.instantiate(config.trainer_args, _convert_="all")

    # training_args = SFTConfig(
    #     do_train=True,
    #     do_eval=True,
    #     output_dir="sft_checkpoints",
    #     evaluation_strategy=config.trainer_args.evaluation_strategy,
    #     save_strategy="epoch",
    #     per_device_train_batch_size=config.trainer_args.per_device_train_batch_size,
    #     per_device_eval_batch_size=config.trainer_args.per_device_eval_batch_size,
    #     num_train_epochs=config.trainer_args.num_train_epochs,
    #     logging_dir=config.trainer_args.logging_dir,
    #     report_to="wandb",
    #     logging_steps=10,
    #     push_to_hub=False,
    #     # push_to_hub=True,
    #     # push_to_hub_model_id=f"{config.model.model_name_or_path.split('/')[-1]}-{config.task.task_name}-{config.task.default_prompt[:10]}",
    # )
    
    collator = DataCollatorForCompletionOnlyLM(tokenizer.encode("Let me solve this step by step.")[2:], tokenizer=tokenizer) #to skip BOS token and the first token sometimes being weirdly generated _Let vs Let_
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks,
        formatting_func=lambda examples: [p + c for p, c in zip(examples["prompt"], examples["completion"])],
        data_collator=collator,
    )

    for callback in callbacks:
        callback.trainer = trainer

    logger.info(f"*** Starting SFT training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ***")
    trainer.train()

    logger.info("*** Final model and tokenizer saving ***")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


@hydra.main(version_base="1.3", config_path="../config", config_name="sft_train_and_pushtohub.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    outputs = trainer(cfg)
    return outputs


if __name__ == "__main__":
    main()
