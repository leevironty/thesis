#!/bin/bash -l

#SBATCH --time=1-0
#SBATCH --mem=12G
#SBATCH --job-name=sweep-4
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=1-10
#SBATCH --mail-type=END,FAIL

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load miniconda
conda activate thesis

srun --gpus=1 --mem=12G  -c 4 wandb agent leevi-ronty/thesis/cj3eqay3
