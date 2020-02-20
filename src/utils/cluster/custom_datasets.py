import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import random


class Gaussian2DDataset(Dataset):
    def __init__(self, samples_per_cluster, num_clusters, std=1, sample_range=10):
        # Initialize dataset properties
        self.samples_per_cluster = samples_per_cluster
        self.num_clusters = num_clusters
        self.std = std

        # Draw cluster centers from 'chessboard'
        cluster_centers = [(i,j) for i in range(sample_range) for j in range(sample_range) if i+j%2 == 0]
        self.locations = random.sample(cluster_centers, self.num_clusters)

        self.data = []
        # Sample around cluster centers
        for loc in self.locations:
            self.data.extend(np.random.normal(loc=loc, scale=std, size=(self.samples_per_cluster,2)))

        self.data = np.array(self.data)
        print('data', self.data.size)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.samples_per_cluster * self.num_clusters