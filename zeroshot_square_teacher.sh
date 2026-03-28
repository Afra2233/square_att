#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --nodes=1

#SBATCH --ntasks=1  
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4

module load anaconda3/2022.05 
source activate square_att
cd /scratch/hpc/07/zhang303/square_att

srun python zeroshot_square_teacher.py --epochs 10 --train-subset 5000 --eval-subset 200 --attack-queries 100