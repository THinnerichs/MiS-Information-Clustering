import numpy as np

import torch
import torchvision

from datasets import *
from transformations import *


def build_dataloaders(config):
    """
    Wrapper for build functions to other datasets.
    :return:                DataLoaders for dataset
    """

    # Initialize dataset
    dataset = None

    if 'MNIST' in config.dataset:
        dataset = torchvision.datasets.MNIST
    elif 'CIFAR100' in config.dataset:
        dataset = torchvision.datasets.CIFAR100
    elif 'CIFAR10' in config.dataset:
        dataset = torchvision.datasets.CIFAR10
    elif 'STL10' in config.dataset:
        dataset = torchvision.datasets.STL10
    elif config.dataset == 'Gaussian2DDataset':
        dataset = Gaussian2DDataset(samples_per_cluster=50,
                                    num_clusters=20,
                                    std=1,
                                    sample_range=10)
    else:
        raise ValueError

    # Initialize transforms
    tf1 = tf2 = tf3 = None
    if config.transformation == 'standard':
        if 'MNIST' in config.dataset:
            tf1, tf2, tf3 = greyscale_make_transforms(config)
        elif 'CIFAR10' in config.dataset:
            tf1, tf2, tf3 = sobel_make_transforms(config)
        elif 'CIFAR100' in config.dataset:
            tf1, tf2, tf3 = sobel_make_transforms(config)
        elif config.dataset == 'Gaussian2DDataset':
            print("Perturbation not yet implemented.")
            raise NotImplementedError
        else:
            raise ValueError
    elif config.transformation == 'linearized_l2':
        if 'MNIST' in config.dataset:
            tf1, tf2, tf3 = greyscale_make_transforms(config)
        elif 'CIFAR10' in config.dataset:
            tf1, tf2, tf3 = sobel_make_transforms(config)
        elif 'CIFAR100' in config.dataset:
            tf1, tf2, tf3 = sobel_make_transforms(config)
        elif config.dataset == 'Gaussian2DDataset':
            print("Perturbation not yet implemented.")
            raise NotImplementedError
        else:
            raise ValueError

