cd /dlabscratch1/amani/LLM-RL/
source /dlabscratch1/amani/miniconda3/bin/activate
conda activate verl

python3 src/train.py experiment=ppo data=gsm8k model_path="meta-llama/Llama-3.2-1B" data.train_dataset_type=adaptive actor_rollout_ref.rollout.gpu_memory_utilization=0.6


