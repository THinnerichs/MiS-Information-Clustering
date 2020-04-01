import numpy as np
import subprocess



preface_script = '''#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J {}
#SBATCH -o ./SLURM_jobs/jobscript_outputs/{}.%J.out
#SBATCH -e ./SLURM_jobs/jobscript_outputs/{}.%J.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=300G
#SBATCH --constraint=[gpu]

#run the application:
module load anaconda3/4.4.0
source /home/${USER}/.bashrc
conda activate ~/.conda/envs/IIC-Clustering/

module load cuda/10.0.130

'''

def run_MNIST_Sinkhorn_job(radius=0.01,
                           batch_size=512,
                           num_sinkhorn_dataloaders=5):
    # "CUDA_VISIBLE_DEVICES=0 " \
    command = "PYTHONPATH='.' " \
              "python3 src/scripts/cluster/cluster_greyscale_twohead_sinkhorn.py " \
              "--model_ind 686 " \
              "--arch ClusterNet6cTwoHead " \
              "--mode IID " \
              "--dataset MNIST " \
              "--dataset_root datasets/MNIST_twohead " \
              "--gt_k 10 " \
              "--output_k_A 50 " \
              "--output_k_B 10  " \
              "--lamb_A 1.0 " \
              "--lamb_B 1.0 " \
              "--lr 0.0001 " \
              "--num_epochs 50 " \
              "--batch_sz 4000 " \
              "--num_dataloaders 5 " \
              "--num_sub_heads 5 " \
              "--num_sinkhorn_dataloaders " + str(num_sinkhorn_dataloaders) + " " \
              "--sinkhorn_batch_size " + str(batch_size) + " " \
              "--sinkhorn_WS_radius " + str(radius)+" "\
              "--crop_orig " \
              "--crop_other " \
              "--tf1_crop centre_half " \
              "--tf2_crop random " \
              "--tf1_crop_sz 20  " \
              "--tf2_crop_szs 16 20 24 " \
              "--input_sz 24 " \
              "--rot_val 25 " \
              "--no_flip " \
              "--head_B_epochs 2 " \
              "--out_root out/MNIST_twohead_Sinkhorn"

    slurm_path = './SLURM_jobs/'
    filename = slurm_path + "Sinkhorn_jobscript.sh"
    with open(file=filename, mode='w') as f:
        f.write(preface_script.format('Sinkhorn', 'Sinkhorn', 'Sinkhorn'))
        f.write(command)

    subprocess.call('sbatch '+filename, shell=True)

def run_MNIST_normal_job():
    command = "CUDA_VISIBLE_DEVICES=0 " \
              "PYTHONPATH='.' " \
              "python3 src/scripts/cluster/cluster_greyscale_twohead.py " \
              "--model_ind 685 " \
              "--arch ClusterNet6cTwoHead " \
              "--mode IID " \
              "--dataset MNIST " \
              "--dataset_root datasets/MNIST_twohead " \
              "--gt_k 10 " \
              "--output_k_A 50 " \
              "--output_k_B 10  " \
              "--lamb_A 1.0 " \
              "--lamb_B 1.0 " \
              "--lr 0.0001 " \
              "--num_epochs 20 " \
              "--batch_sz 4000 " \
              "--num_dataloaders 5 " \
              "--num_sub_heads 5 " \
              "--crop_orig " \
              "--crop_other " \
              "--tf1_crop centre_half " \
              "--tf2_crop random " \
              "--tf1_crop_sz 20  " \
              "--tf2_crop_szs 16 20 24 " \
              "--input_sz 24 " \
              "--rot_val 25 " \
              "--no_flip " \
              "--head_B_epochs 2 " \
              "--out_root out/MNIST_twohead"

    slurm_path = './SLURM_jobs/'
    filename = slurm_path + "NormalMNIST_jobscript.sh"
    with open(file=filename, mode='w') as f:
        f.write(preface_script.format('NormalMNIST', 'NormalMNIST', 'NormalMNIST'))
        f.write(command)

    subprocess.call('sbatch '+filename, shell=True)


if __name__=='__main__':
    run_MNIST_Sinkhorn_job(radius=0.01, batch_size=4096, num_sinkhorn_dataloaders=5)

    run_MNIST_normal_job()
