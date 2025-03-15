# LLM-RL

```bash
conda activate LLM-RL
accelerate launch --num_processes 7 --config_file ./config/deepspeed_zero3.yaml GRPO_GSM8k_curriculum.py --config ./config/grpo-qwen-2.5-05b-gsm8k-uniform.yaml
```

## GRPO GSM8k UNIFORM-CURRICULUM SINGLE-GPU

```bash
conda activate LLM-RL
accelerate launch --num_processes 1 --config_file ./config/accelerate_deepspeed_zero3_config.yaml ./src/train.py experiment=qwen_gsm8k_uniform


python ./src/train.py experiment=qwen_gsm8k_uniform