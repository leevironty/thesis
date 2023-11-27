#!/bin/bash -l

#SBATCH --time=0-8
#SBATCH --mem=4G
#SBATCH --job-name=eval
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/main/%A_%a.log
#SBATCH --error=slurm/main/%A_%a.log


export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load miniconda
conda activate thesis

LOG_PATH=slurm/$SLURM_JOB_NAME\_$SLURM_ARRAY_JOB_ID


srun -c 2 --mem=4G thesis --threads=2 evaluate \
    --dataset solutions/data/nov-24-preference-fix \
    --share=0.1 \
    --time-limit=30 \
    --checkpoint=checkpoints/sweep-4/745dkfpe/checkpoint-epoch=08-val_loss=0.0075.ckpt
