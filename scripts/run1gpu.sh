cd /dlabscratch1/amani/LLM-RL/
source /dlabscratch1/amani/miniconda3/bin/activate
conda activate LLM-RL

# Run the training script
python src/train.py experiment=adaptive model=qwen_1.5b task=gsm8k \
dataset_wrapper=adaptive dataset_wrapper.reward_threshold=0.125 