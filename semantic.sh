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


srun python semantic_score_shaping_square_eval.py \
  --dataset cifar10 \
  --data-root ./data \
  --model ViT-B-32 \
  --pretrained openai \
  --batch-size 64 \
  --num-workers 4 \
  --epsilon 8/255 \
  --square-steps 200 \
  --samples-per-class 100 \
  --top-k 8 \
  --amplitude 0.35 \
  --frequency 1.2