import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def extract_trajectory_stats(attempted_ratios_list):
    """Convert raw trajectory data to padded matrices for portion and reward."""
    num_samples = len(attempted_ratios_list)
    max_iters = max(len(sample) for sample in attempted_ratios_list)

    portion_matrix = np.full((num_samples, max_iters), np.nan)
    reward_matrix = np.full((num_samples, max_iters), np.nan)

    for i, sample_history in enumerate(attempted_ratios_list):
        for t, entry in enumerate(sample_history):
            portion_matrix[i, t] = entry['portion']
            reward_matrix[i, t] = np.mean(entry['reward'])
            if np.isnan(reward_matrix[i, t]):
                reward_matrix[i, t] = reward_matrix[i, t-1]
    return portion_matrix, reward_matrix

def plot_portion_and_reward_trajectories(portion_matrix, reward_matrix, sample_ids=None, path=None):
    """Plot two subplots showing trajectories for portion and reward with mean ± std and selected samples."""
    num_samples, max_iters = portion_matrix.shape
    iterations = np.arange(max_iters)

    # Compute mean and std (ignore NaNs)
    portion_mean = np.nanmean(portion_matrix, axis=0)
    portion_std = np.nanstd(portion_matrix, axis=0)
    reward_mean = np.nanmean(reward_matrix, axis=0)
    reward_std = np.nanstd(reward_matrix, axis=0)

    if sample_ids is None:
        sample_ids = random.sample(range(num_samples), min(10, num_samples))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot portion
    axes[0].plot(iterations, portion_mean, label='Mean Portion', linestyle='--')
    axes[0].fill_between(iterations, portion_mean - portion_std, portion_mean + portion_std, alpha=0.2)
    for sid in sample_ids:
        axes[0].plot(iterations, portion_matrix[sid], alpha=0.7, linewidth=1)
    axes[0].set_ylabel("Supervision Ratio (Portion)")
    axes[0].set_title("Supervision Ratio Over Iterations")
    axes[0].legend()
    axes[0].grid(True)

    # Plot reward
    axes[1].plot(iterations, reward_mean, label='Mean Reward', linestyle='--', color='tab:green')
    axes[1].fill_between(iterations, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2, color='tab:green')
    for sid in sample_ids:
        axes[1].plot(iterations, reward_matrix[sid], alpha=0.7, linewidth=1, color='tab:green')
    axes[1].set_ylabel("Reward")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title("Reward Over Iterations")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=300)
    plt.show()

if __name__ == "__main__":
    # Load state dict
    ratio_actor_path = './LLM-RL/scripts/scratch/ratio_actor-2.pt'
    state_dict = torch.load(ratio_actor_path, weights_only=False)

    attempted_ratios_list = state_dict['attempted_ratios_list']
    portion_matrix, reward_matrix = extract_trajectory_stats(attempted_ratios_list)

    # Save plot in same directory as script
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories_summary.png")
    plot_portion_and_reward_trajectories(portion_matrix, reward_matrix, path=save_path)
