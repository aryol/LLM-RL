export VLLM_LOGGING_LEVEL=ERROR && CUDA_VISIBLE_DEVICES=7 /miniforge/envs/iris/bin/trl \
vllm-serve --model Qwen/Qwen2-0.5B-Instruct > vllm_logs.txt 2>&1 & \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
/miniforge/envs/iris/bin/accelerate launch --num_processes 7 --config_file ./config/deepspeed_zero2.yaml \
src/train.py experiment=qwen_gsm8k model=qwen_0.5b wandb_config.name=Qwen0.5Bdebug trainer_args=vllm \
dataset_wrapper=adaptive dataset_wrapper.reward_threshold=0.125 \
&& kill %1