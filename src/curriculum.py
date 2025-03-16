from transformers import TrainerCallback
import numpy as np
import wandb


class PortionBaseUpdate(TrainerCallback):
    """
    Update the portion of the environment actions that is used for training, linearly according to the current timestep.
    """
    
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset  # Reference to the curriculum dataset

    def update(self, state):
        self.dataset.global_step = state.global_step

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the end of every step to update difficulty dynamically."""
        self.update(state)

    def on_train_begin(self, args, state, control, **kwargs):
        self.update(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        This method is called at the end of every logging step. You can log the dataset's hyperparameters here.
        """
        # Assuming dataset has a 'hyperparameters' attribute (it could also be a method like `self.dataset.get_hyperparameters()`)
        hyperparameters_to_log = self.get_hyperparameters()
        portions = self.dataset.portions
        self.dataset.portions = []

        # Log the hyperparameters to the logs
        if logs is None:
            logs = {}

        if not control.should_evaluate:
            portions = np.array(portions)
            wandb_logs = {'ratio/dataset_portions_mean': portions.mean(), 'ratio/dataset_portions_std': portions.std()}
            wandb_logs.update(hyperparameters_to_log)

            if wandb.run is not None and state.is_world_process_zero:
                wandb.log(wandb_logs, commit=False)
            logs.update(wandb_logs)

    # def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
    #     pass

    # def on_evaluate(self, args, state, control, **kwargs):
    #     pass

    # def on_predict(self, args, state, control, **kwargs):
    #     pass

    # def on_train_end(self, args, state, control, **kwargs):
    #     pass
   


class PortionBetaUpdate(PortionBaseUpdate):
    def __init__(self, dataset, init_alpha, init_beta, final_alpha, final_beta, warmup_timesteps, total_timesteps):
        super().__init__(dataset)
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.final_alpha = final_alpha
        self.final_beta = final_beta
        self.warmup_timesteps = warmup_timesteps
        self.total_timesteps = total_timesteps
        self.hparams = {'ratio/alpha': init_alpha, 'ratio/beta': init_beta}

    def update(self, state):
        """Updates dataset difficulty using a beta distribution based on training progress."""
        super().update(state)
        current_step = state.global_step  # Use HF Trainer’s step counter

        if current_step < self.warmup_timesteps:
            alpha, beta = self.init_alpha, self.init_beta
        elif current_step < self.total_timesteps:
            alpha = self.init_alpha + (self.final_alpha - self.init_alpha) * ((current_step - self.warmup_timesteps) / (self.total_timesteps - self.warmup_timesteps))
            beta = self.init_beta + (self.final_beta - self.init_beta) * ((current_step - self.warmup_timesteps) / (self.total_timesteps - self.warmup_timesteps))
        else:
            alpha, beta = self.final_alpha, self.final_beta
        
        self.hparams =  {'ratio/alpha': alpha, 'ratio/beta': beta}

        # Sample new portion from updated beta distribution
        # Sample new portion from updated uniform distribution
        def sample_ratio_with_seed(seed=42, size=1):
            return np.random.default_rng(seed).beta(alpha, beta, size=size)
        # Update dataset portion dynamically
        self.dataset.set_portion(sample_ratio_with_seed)
    
    def get_hyperparameters(self,):
        return self.hparams


class PortionUniformUpdate(PortionBaseUpdate):
    def __init__(self, dataset, lower_bound_init, lower_bound_final, upper_bound_init, upper_bound_final, warmup_timesteps, total_timesteps):
        super().__init__(dataset)
        self.lower_bound_init = lower_bound_init
        self.lower_bound_final = lower_bound_final
        self.upper_bound_init = upper_bound_init
        self.upper_bound_final = upper_bound_final
        self.warmup_timesteps = warmup_timesteps
        self.total_timesteps = total_timesteps

        self.hparams = {'ratio/lower_bound': lower_bound_init, 'ratio/upper_bound': upper_bound_init}  

    def update(self, state):
        """Updates dataset difficulty using a linear progression based on training progress."""
        super().update(state)
        current_step = state.global_step  # Use HF Trainer’s step counter

        if current_step < self.warmup_timesteps:
            lower_bound, upper_bound = self.lower_bound_init, self.upper_bound_init
        elif current_step < self.total_timesteps:
            lower_bound = self.lower_bound_init + (self.lower_bound_final - self.lower_bound_init) * ((current_step - self.warmup_timesteps) / (self.total_timesteps - self.warmup_timesteps))
            upper_bound = self.upper_bound_init + (self.upper_bound_final - self.upper_bound_init) * ((current_step - self.warmup_timesteps) / (self.total_timesteps - self.warmup_timesteps))
        else:
            lower_bound, upper_bound = self.lower_bound_final, self.upper_bound_final
        
        self.hparams = {'ratio/lower_bound': lower_bound, 'ratio/upper_bound': upper_bound}

        # Sample new portion from updated uniform distribution
        def sample_ratio_with_seed(seed=42, size=1):
            return np.random.default_rng(seed).uniform(low=lower_bound, high=upper_bound, size=size)

        # Update dataset portion dynamically
        self.dataset.set_portion(sample_ratio_with_seed)

    def get_hyperparameters(self,):
        return {'ratio/lower_bound': self.lower_bound_final, 'ratio/upper_bound': self.upper_bound_final}
