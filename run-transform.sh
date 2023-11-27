#!/bin/bash -l

#SBATCH --time=00-01
#SBATCH --mem=8G
#SBATCH --job-name=data-transform
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/main/%A_%a.log
#SBATCH --error=slurm/main/%A_%a.log


export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load miniconda
conda activate thesis


srun -c 8 --mem=2G thesis --threads=8 transform \
    --dataset solutions/data/nov-24-preference-fix 
