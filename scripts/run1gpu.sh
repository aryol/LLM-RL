cd /dlabscratch1/amani/LLM-RL/
source /dlabscratch1/amani/miniconda3/bin/activate
conda activate LLM-RL

# Run the training script
# python src/train.py experiment=adaptive model=tinyllama_1b task=gsm8k dataset_wrapper.reward_threshold=0.125 

python src/train.py experiment=PPO model=gpt2xl_1epochsft task=gsm8k \
wandb_config.name=Qwen0.5Bdebug dataset_wrapper=adaptive dataset_wrapper.reward_threshold=0.12