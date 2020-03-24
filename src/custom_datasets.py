import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import random

<<<<<<< HEAD
from src.transformations import *
import torchvision
=======
import torchvision
from src.transformations import *
>>>>>>> newbranch


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


class Sinkhorn_deformed_Dataset(Dataset):
    def __init__(self, config, device, tf1, tf2, processing_batch_size=128, target_transform=None, radius=0.01):

        print("Building Sinkhorn deformed dataset...")
        # Take images from MNIST and transform each batch with the iterated Sinkhorn attack
        dataset = torchvision.datasets.MNIST(
            root=config.dataset_root,
            transform=tf2,
            target_transform=target_transform,
            download=True)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)


        batch_list = []
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            print("Sinkhorn epoch: {}".format(batch_idx))

            inputs, targets = inputs.to(device), targets.to(device)
            inputs_pgd, _, epsilons = greyscale_sinkhorn_ball_perturbation(inputs,
                                                                           epsilon_factor=1.4,
                                                                           epsilon=radius,
                                                                           maxiters=50,
                                                                           epsilon_iters=5,
                                                                           p=2,
                                                                           regularization=1000,
                                                                           alpha=0.1,
                                                                           norm='wasserstein',
                                                                           ball='wasserstein')

            print(inputs_pgd.size())

            batch_list += inputs_pgd

        self.data = torch.cat(batch_list, 0)
        print("Finished.")

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
<<<<<<< HEAD
=======

>>>>>>> newbranch