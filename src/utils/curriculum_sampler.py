from torch.utils.data import Sampler
import torch
import os
import numpy as np

class CurriculumSampler(Sampler):
    def __init__(self, dataset_size, attempted_ratio_list, epsilon=0.01, easy_floor=0.02):
        """
        Args:
            dataset_size (int): Number of samples.
            attempted_ratios_list (List[List[Dict]]): Past attempts per sample.
            epsilon (float): Minimum sampling weight to prevent collapse.
            easy_floor (float): Force a small minimum probability for easy samples.
        """
        self.dataset_size = dataset_size
        self.attempted_ratio_list = attempted_ratio_list
        self.epsilon = epsilon
        self.easy_floor = easy_floor

    def compute_weights(self):
        weights = []
        for i in range(self.dataset_size):
            history = self.attempted_ratio_list[i]
            if len(history) == 0:
                weights.append(1.0)
            else:
                latest = history[-1]
                portion = latest['portion']
                reward = np.mean(latest['reward'])

                # Base weight: encourage hard samples
                weight = (1.0 - reward) + portion 

                # Easy sample detection
                if reward >= 0.99 and portion <= 0.01:
                    # Very easy sample → clamp to easy_floor
                    weight = self.easy_floor

                # General floor
                weight = max(weight, self.epsilon)
                
                weights.append(weight)

        return torch.tensor(weights)

    def __iter__(self):
        print(f"process id: {os.getpid()} Curriculum Sampler: Sampling with weight, computing weights and building the iterator")
        weights = self.compute_weights()
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=self.dataset_size, replacement=True)
        return iter(sampler)

    def __len__(self):
        return self.dataset_size
