#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J NormalMNIST
#SBATCH -o ./SLURM_jobs/NormalMNIST.%J.out
#SBATCH -e ./SLURM_jobs/NormalMNIST.%J.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=300G
#SBATCH --constraint=[gpu]

#run the application:
module load anaconda3/4.4.0
source /home/hinnertr/.bashrc
conda activate ~/.conda/envs/IIC-Clustering/

module load cuda/10.0.130

CUDA_VISIBLE_DEVICES=0 PYTHONPATH='.' python3 src/scripts/cluster/cluster_greyscale_twohead.py --model_ind 685 --arch ClusterNet6cTwoHead --mode IID --dataset MNIST --dataset_root datasets/MNIST_twohead --gt_k 10 --output_k_A 50 --output_k_B 10  --lamb_A 1.0 --lamb_B 1.0 --lr 0.0001 --num_epochs 20 --batch_sz 4000 --num_dataloaders 5 --num_sub_heads 5 --crop_orig --crop_other --tf1_crop centre_half --tf2_crop random --tf1_crop_sz 20  --tf2_crop_szs 16 20 24 --input_sz 24 --rot_val 25 --no_flip --head_B_epochs 2 --out_root out/MNIST_twohead