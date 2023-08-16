#!/bin/bash -l

#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --job-name=big-data-gen-staging
#SBATCH --array=0-19
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/main/%A_%a.log
#SBATCH --error=slurm/main/%A_%a.log


export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load miniconda
conda activate thesis

LOG_PATH=slurm/$SLURM_JOB_NAME\_$SLURM_ARRAY_JOB_ID

mkdir -p $LOG_PATH

# srun --output="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
#     --error="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
#     poetry run thesis data-gen \
#     -c 10 --od-share 0.3 --activity-drop-prob 0.0 \
#     --seed $SLURM_ARRAY_TASK_ID --out solutions/data/naming-test \
#     --time-limit 120 timpasslib/toy_2

# srun --output="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
#     --error="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
#     poetry run thesis evaluate \
#     --time-limit 600 --baseline timpasslib/grid

srun --output="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
    --error="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
    poetry run thesis data-gen \
    --count 10 \
    --seed $SLURM_ARRAY_TASK_ID --out solutions/data/big-gen-toy-staging \
    --time-limit 40 timpasslib/toy


# srun env | sort


# srun poetry run thesis data-gen -c 50 --od-share 0.015 --activity-drop-prob 0.1 \
#     --seed $SLURM_ARRAY_TASK_ID --out solutions/data/grid \
#     --time-limit 120 timpasslib/grid
