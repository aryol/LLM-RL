cd /dlabscratch1/amani/LLM-RL/
source /dlabscratch1/amani/miniconda3/bin/activate
conda activate verl


set -x

torchrun scripts/sft_on_dataset/train.py \
    data.name=math \
    model.partial_pretrain=meta-llama/Llama-3.2-1B \
    data.micro_batch_size_per_gpu=4 \
    data.train_batch_size=256 \
    trainer.total_epochs=2
    
