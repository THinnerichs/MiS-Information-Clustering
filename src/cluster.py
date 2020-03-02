import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

import argparse
import itertools
import os

from src.utils.cluster.general import config_to_str


"""
  Fully unsupervised clustering ("IIC" = "IID").
  Train and test script (greyscale datasets).
  Network has two heads, for overclustering and final clustering.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model_ind", type=int, required=True)
parser.add_argument("--arch", type=str, default="ClusterNet4h")
parser.add_argument("--opt", type=str, default="Adam")

parser.add_argument("--dataset", type=str, default="MNIST")
parser.add_argument("--dataset_root", type=str,
                    default="/scratch/local/ssd/xuji/MNIST")
parser.add_argument("--gt_k", type=int, default=10)
parser.add_argument("--output_k_A", type=int, required=True)
parser.add_argument("--output_k_B", type=int, required=True)

parser.add_argument("--lamb_A", type=float, default=1.0)
parser.add_argument("--lamb_B", type=float, default=1.0)

parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
parser.add_argument("--lr_mult", type=float, default=0.1)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_sz", type=int, default=240)  # num pairs
parser.add_argument("--num_dataloaders", type=int, default=3)
parser.add_argument("--num_sub_heads", type=int, default=5)

parser.add_argument("--out_root", type=str,
                    default="/scratch/shared/slow/xuji/iid_private")

parser.add_argument("--tf1_crop", type=str, default="random")  # type name
parser.add_argument("--tf2_crop", type=str, default="random")
parser.add_argument("--tf1_crop_sz", type=int, default=84)
parser.add_argument("--tf2_crop_szs", type=int, nargs="+",
                    default=[84])  # allow diff crop for imgs_tf
parser.add_argument("--tf3_crop_diff", dest="tf3_crop_diff", default=False,
                    action="store_true")
parser.add_argument("--tf3_crop_sz", type=int, default=0)
parser.add_argument("--input_sz", type=int, default=96)

parser.add_argument("--rot_val", type=float, default=0.)
parser.add_argument("--always_rot", dest="always_rot", default=False,
                    action="store_true")
parser.add_argument("--no_jitter", dest="no_jitter", default=False,
                    action="store_true")
parser.add_argument("--no_flip", dest="no_flip", default=False,
                    action="store_true")

config = parser.parse_args()

# Build main function in here
config.in_channels = 1
config.out_dir = os.path.join(config.out_root, str(config.model_ind))

assert (config.batch_sz % config.num_dataloaders == 0)

config.dataloader_batch_sz = config.batch_sz / config.num_dataloaders
config.eval_mode = "hung"

# Make out directory if not present
if not os.path.exists(config.out_dir):
  os.makedirs(config.out_dir)

print("Config: {}".format(config_to_str(config)))

def train():
    dataset = None
    dataloaders = DataLoader(dataset,
                             batch_size=config.dataloader_batch_sz,
                             shuffle=True,
                             num_workers=4)


