dataset_path = '/dlabscratch1/amani/LLM-RL/data/verl-data/math/train.parquet'

import datasets

train_dataset = datasets.load_dataset('parquet', data_files=dataset_path, split='train')

print(train_dataset[0])