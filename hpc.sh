#!/bin/bash -l

#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --job-name=demo
#SBATCH --array=1

srun poetry run thesis data-gen -c 3 --od-share 0.1 \
    --seed $SLURM_ARRAY_TASK_ID --out solutions/array-$SLURM_ARRAY_TASK_ID \
    timpasslib/toy_2
