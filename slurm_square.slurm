#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1  
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --cpus-per-task=4

module load anaconda3/2022.05 
source activate square_att
cd /scratch/hpc/07/zhang303/square_att
srun python square_attack.py


