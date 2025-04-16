cd /dlabscratch1/amani/LLM-RL/
source /dlabscratch1/amani/miniconda3/bin/activate
conda activate LLM-RL

# Run the training script
# python src/train.py experiment=GRPO model=gpt2xl_1epochsft task=gsm8k dataset_wrapper=adaptive

# python src/train.py experiment=PPO model=qwen_instr_0.5b task=gsm8k dataset_wrapper=adaptive

# accelerate launch --num_processes 1 --config_file ./config/deepspeed_zero2.yaml \
# src/train.py experiment=GRPO model=qwen_0.5b task=gsm8k wandb_config.name=Qwen0.5Bdebug dataset_wrapper=adaptive

# accelerate launch --num_processes 8 --config_file ./config/deepspeed_zero3.yaml \
# src/train.py experiment=PPO model=qwen_0.5b task=gsm8k wandb_config.name=Qwen0.5Bdebug dataset_wrapper=adaptive


bash run_gemma.sh "trainer.n_gpus_per_node=1 actor_rollout_ref.rollout.tensor_model_parallel_size=1 trainer.logger=['console'] critic.model.path=Qwen/Qwen2.5-0.5B-Instruct actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct data.train_batch_size=256 actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.actor.ppo_micro_batch_size=2 critic.ppo_micro_batch_size=2"
