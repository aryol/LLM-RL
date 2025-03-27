import hydra   
from omegaconf import DictConfig, OmegaConf
from typing import Optional
from omegaconf import OmegaConf
from trl import GRPOConfig
import logging
from datetime import datetime
from typing import Optional
import wandb
from transformers import AutoTokenizer
from functools import partial
from trl import GRPOTrainer, get_peft_config, GRPOConfig
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

from src.utils import get_checkpoint, CurriculumDatasetWrapper, PerSampleCurriculumDatasetWrapper

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

    # Load dataset from Hugging Face Hub or local file
    dataset = hydra.utils.instantiate(config.task.dataset, _convert_="all")
    # take a portion of training data for validation
    train_dataset, val_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42).values()
    train_dataset = dataset["train"].shuffle(seed=42)
    test_dataset = dataset["test"].shuffle(seed=42)

    generate_prompt = hydra.utils.get_method(config.generate_prompt)(config, tokenizer=tokenizer)
    train_dataset = PerSampleCurriculumDatasetWrapper(train_dataset, generate_prompt, initial_portion=0.0, prompt_key=config.task.prompt_key, target_key=config.task.target_key)
    val_dataset = CurriculumDatasetWrapper(val_dataset, generate_prompt, initial_portion=0.0, prompt_key=config.task.prompt_key, target_key=config.task.target_key)
    test_dataset = CurriculumDatasetWrapper(test_dataset, generate_prompt, initial_portion=0.0, prompt_key=config.task.prompt_key, target_key=config.task.target_key)
    # test_dataset = test_dataset.map(lambda x: generate_prompt(x))

    answer_reward_class = hydra.utils.instantiate(config.task.reward_class, dataset=train_dataset)
    answer_reward_func = answer_reward_class.CorrectnessReward
    format_reward_func = hydra.utils.get_method(config.task.format_reward_function)

    training_args = OmegaConf.to_container(config.trainer_args, resolve=True)
    training_args = GRPOConfig(**training_args)
    model_args = hydra.utils.instantiate(config.model.model_config)

    # instantiating callbacks
    callbacks = []
    if config.get('callbacks') is not None:
        for callback_key in config.callbacks.keys():
            callback = config.callbacks[callback_key]
            if 'curriculum' in callback._target_:
                callback = hydra.utils.instantiate(callback, dataset=train_dataset)
            elif 'logger' in callback_key:
                callback = hydra.utils.instantiate(callback, object=answer_reward_class)
            else:
                callback = hydra.utils.instantiate(callback)
            callbacks.append(callback)

    # GRPO Trainer
    trainer = GRPOTrainer(
      model=config.model.model_name_or_path,
      reward_funcs=[format_reward_func, answer_reward_func],
      processing_class=tokenizer,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=val_dataset,
      peft_config=get_peft_config(model_args),
      callbacks=callbacks,
    )

    if trainer.accelerator.is_main_process:
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Training/evaluation parameters {training_args}")
        wandb.init(**OmegaConf.to_container(config.wandb_config, resolve=True))

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

@hydra.main(version_base="1.3", config_path="../config", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    outputs  = trainer(cfg)
    return outputs

if __name__ == "__main__":
    main()
