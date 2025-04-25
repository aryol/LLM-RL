# cd /dlabscratch1/amani/LLM-RL/
# source /dlabscratch1/amani/miniconda3/bin/activate
# conda activate verl

# set -x

# # to train:
# torchrun --nproc_per_node=1 scripts/sft_on_dataset/train.py \
#     data.name=math \
#     data.max_length=3072 \
#     model.partial_pretrain=meta-llama/Llama-3.2-1B \
#     data.micro_batch_size_per_gpu=4 \
#     data.train_batch_size=256 \
#     trainer.total_epochs=2
    
# # to just upload models from the above training to hub:
# torchrun --nproc_per_node=1 scripts/sft_on_dataset/train.py \
#     +just_upload_models_to_hub=True \
#     +path_to_checkpoints_folder=/dlabscratch1/amani/prod/LLM-RL/logs/SFT_for_rl/gsm8k_Llama-2-7b-hf/2025-04-24_20-49-32/checkpoints/