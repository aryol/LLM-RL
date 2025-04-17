from vllm import SamplingParams, LLM
import wandb
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import hydra
import os
from datetime import datetime
# import cupy; cupy.cuda.runtime.getDeviceCount()
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import CurriculumDatasetWrapper


def save_with_timestamp(results, data, base_folder=".", name=""):
    # Create folder with current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(base_folder, f"exp_{name}_{timestamp}")
    os.makedirs(folder_path, exist_ok=True)

    # Save the dictionaries using np.save
    np.save(os.path.join(folder_path, "results.npy"), results)
    np.save(os.path.join(folder_path, "save_data.npy"), data)

    print(f"Saved dictionaries to folder: {folder_path}")


def eval_difficulties(llm, dataset: CurriculumDatasetWrapper, questions, ratios, sampling_params, rewards):
    prompts = []
    gt_answers = []
    for r in ratios:
        dataset.set_portion(r)
        for idx in questions:
            prompts.append(dataset[idx]["prompt"])
            gt_answers.append(dataset[idx]["target"])
    
    completions = llm.generate(prompts, sampling_params)
    # for i in range(len(completions)):
    #     print("QUESTION:")
    #     print(prompts[i])
    #     print("GENERATION:")
    #     print(completions[i].outputs[0].text)
    #     print(f"EXPECTED:", gt_answers[i])
    results_per_reward = {}
    num_completions = len(completions[0].outputs)
    for k, v in rewards.items():
        computed_rewards = []
        for i in range(len(prompts)):
            computed_rewards.append(v(completions=[t.text for t in completions[i].outputs], prompts=[prompts[i]]*num_completions, 
                                      target=[gt_answers[i]]*num_completions, id=[questions[i % len(questions)]] * num_completions, do_log=False, do_update=False))
        results_per_reward[k] = np.array(computed_rewards).reshape(len(ratios), len(questions), -1)
    save_data = {
        'questions': questions, 
        'ratios': ratios, 
        'prompts': prompts, 
        'gt_answers': gt_answers, 
        'completions': completions
    }
    return results_per_reward, save_data


@hydra.main(version_base="1.3", config_path="../config", config_name="eval_ratio_sol_prob.yaml")
def main(config) -> None:


    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            config.model.tokenizer_name_or_path if config.model.get('tokenizer_name_or_path') else config.model.model_name_or_path
        ),
        # revision=config.model.model_config.model_revision,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    dataset = hydra.utils.instantiate(config.task.dataset, _convert_="all")
    dataset = dataset[config.split]

    generate_prompt = hydra.utils.get_method(config.generate_prompt)(config, tokenizer=tokenizer)
    dataset = CurriculumDatasetWrapper(dataset, generate_prompt, initial_portion=0.0, prompt_key=config.task.prompt_key, target_key=config.task.target_key)

    answer_reward_func = hydra.utils.instantiate(config.task.reward_class, dataset=dataset).CorrectnessReward
    format_reward_func = hydra.utils.get_method(config.task.format_reward_function)

    llm = LLM(model=config.model.model_name_or_path, tensor_parallel_size=1)

    sampling_params = hydra.utils.instantiate(config.sampling_params, _convert_="all")

    # questions = list(range(len(dataset))) if config.samples is None else config.samples

    results, save_data = eval_difficulties(llm, dataset = dataset, questions=list(range(len(dataset))), ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], sampling_params=sampling_params, rewards={'answer': answer_reward_func, 'format': format_reward_func})
    save_with_timestamp(results, save_data, base_folder=os.getenv("BOLT_ARTIFACT_DIR", "."), name=config.save_name)
    print("Execution finished.")


    # print(f"Results of {model} model on {dataset} dataset")
    # print(
    #     f"Accuracy = {accuracy:.5f}, invalid answer frequency = {invalid_answer_freq}"
    # )

    # save_dir_path = Path(save_dir)
    # save_dir_path.mkdir(parents=True, exist_ok=True)

    # if save_generated_solutions:
    #     # omitting saving in-context in generated dataset
    #     generated_dataset = create_generated_dataset(
    #         eval_dataset, true_solutions, true_answers, outputs, correct_answers
    #     )
    #     gen_data_path = save_dir_path / "results" / "full_results"
    #     gen_data_path.parent.mkdir(parents=True, exist_ok=True)
    #     generated_dataset.save_to_disk(gen_data_path)

    # if save_wandb:
    #     wandb.login()
    #     run_name = f"model: {model} dataset: {dataset} temp: {temp} shots: {num_shots} seed: {seed}"
    #     config = {
    #         "model": model,
    #         "dataset": dataset,
    #         "save_dir": save_dir,
    #         "seed": seed,
    #         "temp": temp,
    #         "max_tokens": max_tokens,
    #         "n": n,
    #         "num_shots": num_shots,
    #         "add_instruction": add_instruction,
    #         "save_wandb": save_wandb,
    #         "save_generated_solutions": save_generated_solutions,
    #     }
    #     wandb.init(project="Study in Errors", name=run_name, config=config)
    #     wandb.summary["accuracy"] = accuracy
    #     wandb.summary["answer extraction failure rate"] = invalid_answer_freq
    #     if add_instruction:
    #         wandb.summary["instruction"] = instruction

    #     if save_generated_solutions:
    #         artifact = wandb.Artifact("generated_solutions", type="dataset")
    #         artifact.add_dir(str(gen_data_path))
    #         wandb.log_artifact(artifact)
    #     wandb.finish()


if __name__ == "__main__":
    main()
