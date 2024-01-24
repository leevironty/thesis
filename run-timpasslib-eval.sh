#!/bin/bash -l

#SBATCH --time=3-0
#SBATCH --mem=8G
#SBATCH --job-name=timpasslib-eval
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/main/%A_%a.log
#SBATCH --error=slurm/main/%A_%a.log


export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load miniconda
conda activate thesis

LOG_PATH=slurm/$SLURM_JOB_NAME\_$SLURM_ARRAY_JOB_ID

# 6h time limit per eval
srun -c 8 --mem=8G thesis eval-timpasslib --time-limit 21600 --checkpoint best-checkpoint-745dkfpe.ckpt
