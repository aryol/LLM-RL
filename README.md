# LLM-RL

## Installation

```bash
conda create -n LLM-RL python=3.11
conda activate LLM-RL
source ./install.sh
``` 

downloading the data
https://github.com/eric-haibin-lin/verl-data/tree/main

verl git version to install
git checkout 550bbbbffe23bc5450db8ce02b256eb75fbf4129

pip install "triton==3.1.0"


#### to continue training a model (actor critic etc), find the logging directory, date and time, then write the command with configs you prefer, and append `day_time=${now:%Y-%m-%d}_${now:%H-%M-%S}` (or maybe also `log_dir=... `) e.g.
```bash
python ./src/train.py experiment=qwen_gsm8k_uniform day_time="2023-10-12_16-00-00" log_dir=./logs/progressive_rl_ppo_on_gsm8k_Qwen2.5-0.5B-Instruct
```

To continue training from last steps, you need to pass `day_time=2023-10-12_16-00-00` or anything for that matter, also, for your jobs if they get preempted, you can set the day_time in the script so it 
keeps going back to the last step and loading that checkpoint


## preparing the data.

for math:
```bash
python scripts/prepare_dataset/prepare_math_dataset.py --local_dir LLM-RL/data/verl-data/math
python scripts/prepare_dataset/prepare_gsm8k_dataset.py --local_dir LLM-RL/data/verl-data/gsm8k
```

for parity;
```bash
python scripts/prepare_dataset/prepare_parity_dataset.py --length 64 --train_size 999999 --test_size 512 --local_dir data/verl-data
```

## Running commands

first activate the conda environment
```bash
conda activate LLM-RL
```

### Multiple GPUs with accelerate <s>god</s>deepspeed
```bash

2. In `./config/train.yaml`, set 
    ```yaml
    num_gpus: 7
    num_nodes: 1
    use_vllm: true
    vllm_device: "cuda:7"
    ``` 
3. Run following command
```bash
accelerate launch --num_processes 7 --config_file ./config/accelerate_deepspeed_zero3_config.yaml ./src/train.py experiment=qwen_gsm8k_uniform
```

### Single GPU, without accelerate or vllm

1. in `./config/train.yaml`, comment out the `use_vllm`, `vllm_device`, and `vllm_gpu_memory_utilization` fields and set `num_gpus: 1` and `num_nodes: 1`

2. Run following command
```bash
python ./src/train.py experiment=qwen_gsm8k_uniform
```


### Single GPU, with accelerate

set 

```bash
accelerate launch --num_processes 1 --config_file ./config/accelerate_deepspeed_zero3.yaml ./src/train.py experiment=qwen_gsm8k_uniform
```

## Downloading data

You need to download JSON files for the datasets you want to use and put them under .data/ directory.
### MATH dataset

You can download the the data here (it's just the MATH dataset in a special format and gsm8k): 
`https://drive.google.com/file/d/1kRv4X3ZDlKj9-4Rf5E5MqzqQWJooX340/view?usp=sharing`

from googledrive it's trick
```bash
cd ./data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kRv4X3ZDlKj9-4Rf5E5MqzqQWJooX340' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kRv4X3ZDlKj9-4Rf5E5MqzqQWJooX340" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip
mv ./data/MATH_json ./MATH
rm data.zip
rm -rf ./data
```

It's also publicly available here but needs to be processed. Each question is one single JSON file...
```bash
cd ./data
wget https://www.modelscope.cn/datasets/opencompass/competition_math/resolve/master/data/MATH.zip 
unzip MATH.zip
```
