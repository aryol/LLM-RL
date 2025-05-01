pip install rootutils
pip uninstall tensordict -y # to get rid of 0.1.2 for god's sake
python -c "import tensordict; print (tensordict.__version__)" # the correct remaining one should be 0.7.0+4018b8b

python3 src/train.py experiment=grpo data.sampler=null data=parity model_path="masani/SFT_parity_Qwen2-0.5B_epoch_5_global_step_15" data.train_dataset_type=adaptive trainer.n_gpus_per_node=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=3

