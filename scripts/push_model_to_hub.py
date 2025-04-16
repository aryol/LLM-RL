from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import create_repo, HfApi

paths = ['/dlabscratch1/amani/LLM-RL/logs/sft_pretrain_and_pushtohub/gsm8k-gpt2-xl/2025-04-15_09-40-47/epoch1']
username = HfApi().whoami()["name"]

for path in paths:
    model_name = "gsm8k-gpt2-xl_1epoch_2025-04-15_09-40-47-witheos"
    path.split('/')[-3:]
    repo_id = f"{username}/{model_name}"
    print(f"Pushed to: {repo_id}")
    create_repo(repo_id, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
