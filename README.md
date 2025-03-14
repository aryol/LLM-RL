# LLM-RL

```bash
conda activate LLM-RL
accelerate launch --num_processes 1 --config_file ./config/deepspeed_zero3.yaml GRPO_GSM8k_curriculum.py --config ./config/grpo-qwen-2.5-05b-gsm8k-uniform.yaml
```
grpo-qwen-2.5-05b-gsm8k-uniform