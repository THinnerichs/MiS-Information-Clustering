import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import random

from src.transformations import *
import torchvision
import torchvision.transforms as transforms

import pickle
import os
import sys


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


class Sinkhorn_deformed_MNIST_Dataset(Dataset):
    def __init__(self, config, device, tf1, tf2, target_transform=None, processing_batch_size=128, radius=0.01, dataloader_num=-1):
        print("Building Sinkhorn deformed dataset...")
        # Take images from MNIST and transform each batch with the iterated Sinkhorn attack
        transform_train = []
        if target_transform:
            transform_train+= tf2
        transform_train += [transforms.ToTensor()]
        transform_train = transforms.Compose(transform_train)


        self.num_classes = 10
        dataset = torchvision.datasets.MNIST(
            root=config.dataset_root,
            transform=transform_train,
            train=True,
            download=True)

        precomputed_path = '../datasets/MNIST_Sinkhorn/'
        filename = 'Sinkhorned_MNIST_' + str(radius) + '_radius_' + str(dataloader_num) + '_dataloader'

        file = precomputed_path + filename + '.pkl'
        if os.path.exists(file):
            with open(file=file, mode='rb') as f:
                self.data = torch.load(f)
        else:
            print('Building Sinkhorn data...')
            trainloader = torch.utils.data.DataLoader(dataset, batch_size=processing_batch_size, shuffle=False, num_workers=1)

            deformed_input_batch_list = []
            targets_list = []

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                print("Sinkhorn batch: {}/{}".format(batch_idx, len(trainloader)))

                inputs, targets = inputs.to(device), targets.to(device)

                inputs_pgd, _, epsilons = greyscale_sinkhorn_ball_perturbation(inputs,
                                                                               num_classes=self.num_classes,
                                                                               device=device,
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

                inputs_pgd = inputs_pgd.view(inputs.size())

                deformed_input_batch_list+= list(inputs_pgd)
                targets_list += list(targets)

            self.data = torch.cat(deformed_input_batch_list, 0)
            self.targets = targets_list

            print('Saving data...')
            with open(file=file, mode='rb') as f:
                torch.save(self.data, f)

        print("Finished.")

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)
