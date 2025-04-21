cd /dlabscratch1/amani/LLM-RL/
source /dlabscratch1/amani/miniconda3/bin/activate
conda activate verl

python3 src/train.py experiment=ppo data=math model_path="Qwen/Qwen2.5-0.5B-Instruct" data.train_dataset_type=adaptive


