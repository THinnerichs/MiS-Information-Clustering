import numpy as np

import torch
import torchvision

from datasets import *


def build_dataloaders(config):
    '''
    Wrapper for build functions to other datasets.
    :return:                DataLoaders for dataset
    '''

    tf1, tf2, tf3 = grey

    dataset = None
    if 'MNIST' in config.dataset:
        dataset = torchvision.datasets.MNIST
    elif 'CIFAR10' in config.dataset:
        dataset = torchvision.datasets.CIFAR10
    elif 'CIFAR100' in config.dataset:
        dataset = torchvision.datasets.CIFAR100
    elif config.dataset == 'Gaussian2DDataset':
        dataset = Gaussian2DDataset(samples_per_cluster=50,
                                    num_clusters=20,
                                    std=1,
                                    sample_range=10)
    else:
        raise ValueError
