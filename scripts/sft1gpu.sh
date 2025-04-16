cd /dlabscratch1/amani/LLM-RL/
source /dlabscratch1/amani/miniconda3/bin/activate
conda activate LLM-RL

# Run the training script
python scripts/sft_pretrain_and_pushtohub.py model=gpt2xl_base task=gsm8k