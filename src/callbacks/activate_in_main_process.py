from transformers import TrainerCallback

class ActivateInMainProcess(TrainerCallback):
    def __init__(self, object):
        """
        Args:
        """
        super().__init__()
        self.object = object

    def on_init_end(self, args, state, control, **kwargs):
        """Initialize log file paths and create necessary directories."""
        if state.is_world_process_zero:
            self.object.is_main_process = True
            print('set shod lol')
        else:
            self.object.is_main_process = False


# code that didn't work 
# import json
# import os
# import torch

# class RolloutLogger(TrainerCallback):
#     """
#     Logs model generations during training and evaluation without interfering with the data loader or model gradients.
#     """

#     def __init__(self, log_every_n_steps=2, num_eval_batches=2):
#         """
#         Args:
#             log_every_n_steps (int): Log one training batch every N steps.
#             num_eval_batches (int): Number of batches to sample predictions from during evaluation.
#         """
#         super().__init__()
#         self.log_every_n_steps = log_every_n_steps
#         self.num_eval_batches = num_eval_batches  # Control eval sampling

#     def on_init_end(self, args, state, control, **kwargs):
#         """Initialize log file paths and create necessary directories."""
#         self.experiment_path = args.output_dir
#         os.makedirs(os.path.join(self.experiment_path, "completion_samples"), exist_ok=True)

#         self.train_log_file = os.path.join(self.experiment_path, "completion_samples", "train_completion_samples.json")
#         self.eval_log_file = os.path.join(self.experiment_path, "completion_samples", "eval_completion_samples.json")

#         # Ensure log files exist
#         for file in [self.train_log_file, self.eval_log_file]:
#             if not os.path.exists(file):
#                 with open(file, "w") as f:
#                     json.dump([], f)  # Initialize as empty list

#     def on_evaluate(self, args, state, control, **kwargs):
#         """Samples `num_eval_batches` from the evaluation dataset and logs model outputs."""
#         trainer = kwargs["trainer"]
#         eval_dataloader = kwargs["eval_dataloader"]
#         tokenizer = trainer.tokenizer

#         sampled_outputs = []
#         model = trainer.model

#         model.eval()  # Ensure eval mode
#         with torch.no_grad():  # Avoid gradient accumulation
#             for i, batch in enumerate(eval_dataloader):
#                 if i >= self.num_eval_batches:
#                     break  # Stop after the set number of batches

#                 inputs = batch["input_ids"].to(model.device)
#                 outputs = model.generate(inputs)  # Generate outputs
                
#                 # Decode for readability
#                 decoded_inputs = [tokenizer.decode(i, skip_special_tokens=True) for i in inputs]
#                 decoded_outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

#                 sampled_outputs.extend(
#                     [{"input": inp, "generated_output": out} for inp, out in zip(decoded_inputs, decoded_outputs)]
#                 )

#         # Append to eval log file
#         with open(self.eval_log_file, "r+") as f:
#             logs = json.load(f)
#             logs.append({"step": state.global_step, "samples": sampled_outputs})
#             f.seek(0)
#             json.dump(logs, f, indent=4)

#     def on_log(self, args, state, control, logs=None, **kwargs):
#         """Logs one batch of training completions every `log_every_n_steps` steps."""
#         if state.global_step % self.log_every_n_steps != 0:
#             return  # Skip if not at logging step

#         model = kwargs["model"]
#         tokenizer = kwargs['processing_class']


#         sampled_outputs = []

#         model.eval()  # Ensure eval mode
#         with torch.no_grad():  # Avoid gradient accumulation
#             for batch in train_dataloader:
#                 inputs = batch["input_ids"].to(model.device)
#                 outputs = model.generate(inputs)  # Generate outputs
                
#                 # Decode for readability
#                 decoded_inputs = [tokenizer.decode(i, skip_special_tokens=True) for i in inputs]
#                 decoded_outputs = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

#                 sampled_outputs.extend(
#                     [{"input": inp, "generated_output": out} for inp, out in zip(decoded_inputs, decoded_outputs)]
#                 )
#                 break  # Only log one batch per log step

#         # Append to train log file
#         with open(self.train_log_file, "r+") as f:
#             logs = json.load(f)
#             logs.append({"step": state.global_step, "samples": sampled_outputs})
#             f.seek(0)
#             json.dump(logs, f, indent=4)
