pip install rootutils
pip uninstall tensordict -y # to get rid of 0.1.2 for god's sake
# python -c "import tensordict; print (tensordict.__version__)" # the correct remaining one should be 0.7.0+4018b8b
python3 src/train.py experiment=ppo data=gsm8k model_path="meta-llama/Llama-2-7b" data.train_dataset_type=adaptive trainer.n_gpus_per_node=4


