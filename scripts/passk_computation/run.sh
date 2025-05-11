

python passk_compute.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path LLM-RL/data/verl-data/modgsm8k \
    --reward_fn None \
    --n_max 20 \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_tokens 1024



python passk_compute.py \
    --model masani/SFT_cumulative_parity_length_16_bitwidth_1_2048_512_Qwen2-1.5B_epoch_8_global_step_64 \
    --dataset_path LLM-RL/data/verl-data/cumulative_parity_length_16_bitwidth_1_1024_512 \
    --reward_fn src.utils.rewards.CumulativeParityReward \
    --n_max 20 \
    --temperature 0.8 \
    --top_p 0.95 \
    --max_tokens 1024
    --save_dir LLM-RL/scripts/passk_computation/outputs



#  to plot
python plot_passk.py \
  --paths outputs/gsm8k--base--eval.jsonl outputs/gsm8k--rl--eval.jsonl outputs/gsm8k--curriculum--eval.jsonl \
  --labels base RL curriculum \
  --kmax 20 \
  --out passk_gsm8k_llama3_1_8b_comparison.png
