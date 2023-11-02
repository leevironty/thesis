#!/bin/bash -l

#SBATCH --time=03:45:00
#SBATCH --partition=gpushort
#SBATCH --mem=12G
#SBATCH --job-name=sweep-2
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load miniconda
conda activate thesis

srun wandb agent leevi-ronty/thesis/5dp1s5pc
