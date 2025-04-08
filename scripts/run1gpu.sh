cd /dlabscratch1/amani/LLM-RL/
source /dlabscratch1/amani/miniconda3/bin/activate
conda activate LLM-RL

# Run the training script
# python src/train.py experiment=GRPO model=gpt2xl_1epochsft task=gsm8k dataset_wrapper=adaptive

# python src/train.py experiment=PPO model=qwen_instr_0.5b task=gsm8k dataset_wrapper=adaptive

# accelerate launch --num_processes 1 --config_file ./config/deepspeed_zero2.yaml \
# src/train.py experiment=GRPO model=qwen_0.5b task=gsm8k wandb_config.name=Qwen0.5Bdebug dataset_wrapper=adaptive

accelerate launch --num_processes 1 --config_file ./config/deepspeed_zero2.yaml src/train.py experiment=PPO model=qwen_0.5b task=gsm8k wandb_config.name=Qwen0.5Bdebug dataset_wrapper=adaptive