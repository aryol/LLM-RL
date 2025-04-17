
# # python3 verl/train.py \
# #     algorithm.adv_estimator=grpo \
# #     data.train_files="$train_files" \
# #     data.val_files="$test_files" \
# #     data.train_batch_size=32 \
# #     data.max_prompt_length=400 \
# #     data.max_response_length=400 \
# #     data.filter_overlong_prompts=True \
# #     data.truncation='error' \
# #     actor_rollout_ref.model.path=Qwen/Qwen2-0.5B-Instruct \
# #     actor_rollout_ref.actor.optim.lr=1e-6 \
# #     actor_rollout_ref.model.use_remove_padding=True \
# #     actor_rollout_ref.actor.ppo_mini_batch_size=32 \
# #     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
# #     actor_rollout_ref.actor.use_kl_loss=True \
# #     actor_rollout_ref.actor.kl_loss_coef=0.001 \
# #     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
# #     actor_rollout_ref.actor.entropy_coeff=0 \
# #     actor_rollout_ref.model.enable_gradient_checkpointing=True \
# #     actor_rollout_ref.actor.fsdp_config.param_offload=False \
# #     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
# #     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
# #     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
# #     actor_rollout_ref.rollout.name=vllm \
# #     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
# #     actor_rollout_ref.rollout.n=5 \
# #     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
# #     actor_rollout_ref.ref.fsdp_config.param_offload=True \
# #     algorithm.use_kl_in_reward=False \
# #     trainer.critic_warmup=0 \
# #     trainer.logger=['console','wandb'] \
# #     trainer.project_name='verl_grpo_example_gsm8k' \
# #     trainer.experiment_name='qwen2_7b_function_rm' \
# #     trainer.n_gpus_per_node=1 \
# #     trainer.nnodes=1 \
# #     trainer.save_freq=-1 \
# #     trainer.test_freq=5 \
# #     trainer.total_epochs=15




# python3 verl/train.py \
#     algorithm.adv_estimator=grpo \
#     data.train_files="$train_files" \
#     data.val_files="$test_files" \
#     data.train_batch_size=32 \
#     data.max_prompt_length=400 \
#     data.max_response_length=400 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=Qwen/Qwen2-0.5B-Instruct \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=32 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.n=5 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name='verl_grpo_example_gsm8k' \
#     trainer.experiment_name='qwen2_7b_function_rm' \
#     trainer.n_gpus_per_node=1 \
#     trainer.nnodes=1 \
#     trainer.save_freq=-1 \
#     trainer.test_freq=5 \
#     trainer.total_epochs=15

cd /dlabscratch1/amani/LLM-RL/
source /dlabscratch1/amani/miniconda3/bin/activate
conda activate verl

python3 src/train.py experiment=ppo data=gsm8k model_path="Qwen/Qwen2.5-0.5B-Instruct"


