
import argparse
import os
import math

import datasets

from verl.utils.hdfs_io import copy, makedirs

def build_r3_split(
    hf_split,
    split_name: str,
    k: int,
):
    """
    Turn a HF split (train/test) into an R3-style list of examples.

    For each original example i, we produce:
      - 1 "base" example (no partial fields)
      - k "partial" examples with increasing prefixes of the solution
    """
    r3_examples = []
    idx = 0
    # {
    #             "data_source": data_source,
    #             "prompt": [
    #                 {
    #                     "role": "user",
    #                     "content": question,
    #                 }
    #             ],
    #             # Full CoT solution
    #             "answer": full_answer,
    #             "ability": "math",
    #             "reward_model": {
    #                 "style": "rule",
    #                 # Only the final boxed answer as ground truth
    #                 "ground_truth": ground_truth,
    #             },
    #             "extra_info": {
    #                 "split": split_name,
    #                 "index": idx,
    #                 "orig_index": original_idx,
    #                 "question": question_raw,
    #             },
    #         }

    for original_idx, example in enumerate(hf_split):
        # Tokenize the full CoT answer at the word level
        full_answer = example["answer"].strip()
        words = full_answer.split()
        n_words = len(words)

        # --- 1) Base example (no partial info) ---
        r3_examples.append(
            {
                "data_source": example["data_source"],
                "prompt": example["prompt"],
                # Full CoT solution
                "answer": full_answer,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    # Only the final boxed answer as ground truth
                    "ground_truth": example['reward_model']["ground_truth"],
                },
                "extra_info": {
                    "split": example['extra_info']['split'],
                    "index": idx,
                    "orig_index": example['extra_info']['index'],
                    "question": example['extra_info']["question"],
                },
            }
        )

        # --- 2) k partial-rationale curriculum variants ---
        # j = 1..k: reveal j/k of the words of the solution
        if n_words > 0:
            for j in range(1, k + 1):
                cutoff = math.ceil(j * n_words / k)
                partial_words = words[:cutoff]
                completion_words = words[cutoff:]

                partial_answer = " ".join(partial_words)
                completion = " ".join(completion_words)
                portion = cutoff / n_words
                idx += 1
                r3_examples.append(
                    {
                        "data_source": example["data_source"],
                        "prompt": example["prompt"],
                        # Still keep the *full* solution as the model target
                        "answer": full_answer,
                        "ability": "math",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": example['reward_model']["ground_truth"],
                        },
                        "extra_info": {
                            "split": split_name,
                            "index": idx,
                            "orig_index": original_idx,
                            "question": example['extra_info']['question'],
                            "partial_answer": partial_answer,
                            "completion": completion,
                            "portion": portion,  # fraction of words revealed
                        },
                    }
                )
                

    return datasets.Dataset.from_list(r3_examples)


if __name__ == "__main__":
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))  # Or another port
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_train_path",
        default="./data/verl-data/DeepScaleR/train.parquet",
        help="Local directory to save parquet files.",
    )

    parser.add_argument(
        "--data_test_path",
        default="./data/verl-data/DeepScaleR/test.parquet",
        help="Local directory to save parquet files.",
    )

    parser.add_argument(
        "--local_dir",
        default="./data/verl-data/deepscaler-r3",
        help="Local directory to save parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy the processed data.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of equal word-level chunks for partial solutions.",
    )
    args = parser.parse_args()
    data_train_path = os.path.expanduser(args.data_train_path)
    data_test_path = os.path.expanduser(args.data_test_path)
    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    k = args.k


    print(f"Loading the {data_train_path} ...", flush=True)
    dataset = datasets.load_dataset('parquet', data_files=data_train_path, trust_remote_code=True)
    train_dataset = dataset["train"]
    print(f"Loading the {data_test_path} ...", flush=True)
    dataset = datasets.load_dataset('parquet', data_files=data_test_path, trust_remote_code=True)   
    test_dataset = dataset["train"]

    print(f"Building R3-style train split with k={k}...", flush=True)
    r3_train_dataset = build_r3_split(
        hf_split=train_dataset,
        split_name="train",
        k=k,
    )

    print(f"Building R3-style test split with k={k}...", flush=True)
    r3_test_dataset = build_r3_split(
        hf_split=test_dataset,
        split_name="test",
        k=0,
    )

    os.makedirs(local_dir, exist_ok=True)
    train_path = os.path.join(local_dir, f"train_k{k}.parquet")
    test_path = os.path.join(local_dir, f"test_k{k}.parquet")

    print(f"Saving train to {train_path}")
    r3_train_dataset.to_parquet(train_path)

    print(f"Saving test to {test_path}")
    r3_test_dataset.to_parquet(test_path)

    if hdfs_dir is not None:
        print(f"Copying to HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print("Copy to HDFS completed.")


# for openr1-math-220k`;
# # python scripts/prepare_dataset/prepare_R3_parquet.py --data_train_path data/verl-data/openr1-math-220k/train.parquet --data_test_path data/verl-data/openr1-math-220k/test.parquet --local_dir ./data/verl-data/openr1-math-r3 --k 4`