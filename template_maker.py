import yaml

def get_model_name(model):
   with open(f'config/model/{model}.yaml', 'r') as f:
      data = yaml.safe_load(f)
      return data['model_name_or_path'], data['model_name']

# counter = 0
# base_counter = 20
# for model in ['gpt2xl_1.5b', 'phi4_4b', 'qwen_0.5b', 'qwen_1.5b', 'qwen_7b', 'qwen_instr_0.5b', 'qwen_instr_1.5b', 'qwen_instr_7b']:
#   for gt in [0.125, 0.250, 0.5, 0.625, 0.875]:
#     for dataset in ['gsm8k', 'math']:
#         completion_len = "" if dataset == 'gsm8k' else "trainer_args.max_completion_length=1000"
#         template = f'''- id: {base_counter + counter}
#   name: "{get_model_name(model)[1]}-{dataset}-adaptive-{gt}-id-{base_counter + counter}"
#   model: "{get_model_name(model)[0]}"
#   command: "accelerate launch --num_processes 7 --config_file ./config/deepspeed_zero2.yaml src/train.py experiment=adaptive trainer_args=vllm dataset_wrapper.reward_threshold={gt} model={model} task={dataset} {completion_len}"'''
#         counter += 1
#         print(template)


# counter = 0
# base_counter = 100
# for model in ['gpt2xl_1.5b', 'phi4_4b', 'qwen_0.5b', 'qwen_1.5b', 'qwen_7b', 'qwen_instr_0.5b', 'qwen_instr_1.5b', 'qwen_instr_7b']:
#     for dataset in ['gsm8k', 'math']:
#         completion_len = "" if dataset == 'gsm8k' else "trainer_args.max_completion_length=1000"
#         template = f'''- id: {base_counter + counter}
#   name: "{get_model_name(model)[1]}-{dataset}-baseline-id-{base_counter + counter}"
#   model: "{get_model_name(model)[0]}"
#   command: "accelerate launch --num_processes 7 --config_file ./config/deepspeed_zero2.yaml src/train.py experiment=default trainer_args=vllm model={model} task={dataset} {completion_len}"'''
#         counter += 1
#         print(template)

# print(" ".join(map(str, range(20, 100))))


# PPO - 12 April 2025

counter = 0
base_counter = 200
for model in ['qwen_0.5b', 'qwen_1.5b', 'qwen_instr_0.5b', 'qwen_instr_1.5b']:
    for dataset in ['gsm8k', 'math']:
      #   completion_len = "" if dataset == 'gsm8k' else "trainer_args.max_completion_length=1000"
        template = f'''- id: {base_counter + counter}
  name: "{get_model_name(model)[1]}-{dataset}-PPO-baseline-id-{base_counter + counter}"
  model: "{get_model_name(model)[0]}"
  command: "accelerate launch --num_processes 8 --config_file ./config/deepspeed_zero2.yaml src/train.py experiment=PPO_{model.split('_')[-1]}_L{512 if dataset=='gsm8k' else 1024} model={model} task={dataset} dataset_wrapper=default"'''
        counter += 1
        print(template)

        template = f'''- id: {base_counter + counter}
  name: "{get_model_name(model)[1]}-{dataset}-PPO-adaptive-0.5-id-{base_counter + counter}"
  model: "{get_model_name(model)[0]}"
  command: "accelerate launch --num_processes 8 --config_file ./config/deepspeed_zero2.yaml src/train.py experiment=PPO_{model.split('_')[-1]}_L{512 if dataset=='gsm8k' else 1024} model={model} task={dataset} dataset_wrapper=adaptive dataset_wrapper.reward_threshold=0.5"'''
        counter += 1
        print(template)


# print(' '.join(list(map(str, range(202, 228)))))