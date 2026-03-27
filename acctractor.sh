#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --nodes=1

#SBATCH --ntasks=1  
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4

module load anaconda3/2022.05 
source activate square_att
cd /scratch/hpc/07/zhang303/square_att
srun python clip_cifar10_anchor_defense_stratified.py \
    --data_root ./data \
    --batch_size 64 \
    --samples_per_class 100 \
    --eps 8 \
    --n_queries 1000 \
    --alpha 0.03 \
    --noise_std 0.01 \
    --topk_pull 1