cd /dlabscratch1/amani/LLM-RL/
source /dlabscratch1/amani/miniconda3/bin/activate
conda activate verl

python3 src/train.py experiment=grpo data=parity model_path="masani/SFT_parity_Qwen2-0.5B_epoch_5_global_step_20" data.train_dataset_type=adaptive \
    data.train_batch_size=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2