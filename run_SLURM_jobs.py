import numpy as np
import subprocess



preface_script = '''#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J {name}
#SBATCH -o ./SLURM_jobs/{name}.%J.out
#SBATCH -e ./SLURM_jobs/{name}.%J.err
#SBATCH --time={days}-00:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=140G
#SBATCH --constraint=[gpu]
#SBATCH --cpus-per-gpu=8

#run the application:
module load anaconda3/4.4.0
source /home/hinnertr/.bashrc
conda activate ~/.conda/envs/IIC-Clustering/

module load cuda/10.0.130

'''

def run_MNIST_Sinkhorn_job(radius=0.01,
                           sinkhorn_batch_size=512,
                           num_sinkhorn_dataloaders=5,
                           epochs=50,
                           identifier=686,
                           days=3):
    # "CUDA_VISIBLE_DEVICES=0 " \
    command = "PYTHONPATH='.' python3 src/scripts/cluster/cluster_greyscale_twohead_sinkhorn.py --model_ind {identifier} --arch ClusterNet6cTwoHead --mode IID --dataset MNIST --dataset_root datasets/MNIST_twohead --gt_k 10 --output_k_A 50 --output_k_B 10 --lamb_A 1.0 --lamb_B 1.0 --lr 0.0001 --num_epochs {num_epochs} --batch_sz 8000 --num_dataloaders 5 --num_sub_heads 5 --num_sinkhorn_dataloaders {num_sinkhorn_dataloaders} --sinkhorn_batch_size {sinkhorn_batch_size} --sinkhorn_WS_radius {WS_radius} --crop_orig --crop_other --tf1_crop centre_half --tf2_crop random --tf1_crop_sz 20 --tf2_crop_szs 16 20 24 --input_sz 24 --rot_val 25 --no_flip --head_B_epochs 2 --out_root out/MNIST_twohead_Sinkhorn".format(identifier=str(identifier),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        num_epochs=str(epochs),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        num_sinkhorn_dataloaders=num_sinkhorn_dataloaders,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        sinkhorn_batch_size=sinkhorn_batch_size,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        WS_radius = radius)


    slurm_path = './SLURM_jobs/'
    filename = slurm_path + "Sinkhorn_jobscript.sh"
    with open(file=filename, mode='w') as f:
        f.write(preface_script.format(name='Sinkhorn', days=str(days)))
        f.write(command)

    subprocess.call('sbatch '+filename, shell=True)

def run_MNIST_normal_job(identifier=685,
                         epochs=50,
                         days=3):
    command = "CUDA_VISIBLE_DEVICES=0 " \
              "PYTHONPATH='.' " \
              "python3 src/scripts/cluster/cluster_greyscale_twohead.py " \
              "--model_ind {identifier} " \
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
              "--num_epochs {num_epochs} " \
              "--batch_sz 8000 " \
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
              "--out_root out/MNIST_twohead".format(identifier=str(identifier),
                                                    num_epochs=epochs)

    slurm_path = './SLURM_jobs/'
    filename = slurm_path + "NormalMNIST_jobscript.sh"
    with open(file=filename, mode='w') as f:
        f.write(preface_script.format(name='NormalMNIST', days=str(days)))
        f.write(command)

    subprocess.call('sbatch '+filename, shell=True)


if __name__=='__main__':
    run_MNIST_Sinkhorn_job(radius=0.01, sinkhorn_batch_size=16384, num_sinkhorn_dataloaders=5, days=1)
    run_MNIST_Sinkhorn_job(radius=0.1, sinkhorn_batch_size=16384, num_sinkhorn_dataloaders=5, days=1)
    run_MNIST_Sinkhorn_job(radius=0.001, sinkhorn_batch_size=16384, num_sinkhorn_dataloaders=5, days=1)
    run_MNIST_Sinkhorn_job(radius=0.2, sinkhorn_batch_size=16384, num_sinkhorn_dataloaders=5, days=1)
    run_MNIST_Sinkhorn_job(radius=1, sinkhorn_batch_size=16384, num_sinkhorn_dataloaders=5, days=1)


    # run_MNIST_normal_job(days=2)
