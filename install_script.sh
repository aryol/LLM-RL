# Install Pytorch & other libraries, make sure to match your GPU driver version
# conda create -n LLM-RL python=3.11
# conda activate LLM-RL

pip install -U tensorboard "setuptools<71.0.0"
 
# Install flash-attn
pip install -U flash-attn 
 
# Install Hugging Face libraries
pip install  --upgrade \
  "transformers==4.48.1" \
  "datasets==3.1.0" \
  "accelerate==1.3.0" \
  "hf-transfer==0.1.9" \
  "deepspeed==0.15.4" \
  "hydra-core==1.3.2" \
  "trl==0.16.0"

pip uninstall -y pynvml
pip install -U nvidia-ml-py
pip install wandb
pip install rootutils

# install vLLM 
pip install -U vllm
# pip install git+https://github.com/huggingface/trl.git
 
## IMPORTANT: If you want to run the notebook and the interactive cells you also need to install the following libraries:
# But first read it the blog post and then decide as they might conflict with the libraries for distributed training. 
# %pip install "peft==0.14.0" "bitsandbytes==0.45.0"
 