#!/bin/bash -l

#SBATCH --time=1-0
#SBATCH --mem=4G
#SBATCH --job-name=data-gen-fixed
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm/main/%A_%a.log
#SBATCH --error=slurm/main/%A_%a.log


export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load miniconda
conda activate thesis

LOG_PATH=slurm/$SLURM_JOB_NAME\_$SLURM_ARRAY_JOB_ID

# mkdir -p $LOG_PATH

# srun -c 16 --mem=16G thesis --threads=16 big-eval \
#     --checkpoint=checkpoint-pref-order.ckpt \
#     --dataset-path=timpasslib/grid \
#     --time-limit=80000 \
#     --rel-gap=0.05


# srun -c 4 --mem=4G thesis --threads=4 data-gen \
#     --seed $SLURM_ARRAY_TASK_ID \
#     --out solutions/correct \
#     --count 2500 --time-limit 30 --preprocess asdf

srun -c 2 --mem=4G thesis --threads=2 big-eval \
    --dataset-path timpasslib/grid \
    --time-limit-trivial=3600 \
    --time-limit-gnn=82800 \
    --rel-gap-trivial=0.05 \
    --rel-gap-gnn=0.15 \
    --checkpoint=checkpoints/sweep-3/4zpgbuf3/checkpoint-epoch=04-val_loss=0.0109.ckpt


# srun -c 2 --mem=2G thesis --threads=2 multi-solution-generation \
#     --seed $SLURM_ARRAY_TASK_ID --out solutions/data/correct-floats \
#     --time-limit 20 --count=400 --alt-solutions=10

# srun --output="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
#     --error="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
#     poetry run thesis evaluate \
#     --time-limit 600 --baseline timpasslib/grid

# srun --output="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
#     --error="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
#     poetry run thesis data-gen \
#     --count 10 \
#     --seed $SLURM_ARRAY_TASK_ID --out solutions/env-test \
#     --time-limit 40 timpasslib/toy

# srun -p gpushort --gres=gpu:1 --output="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
#     --error="$LOG_PATH/$SLURM_ARRAY_TASK_ID.log" \
#     poetry run thesis train

# srun -p gpushort --gres=gpu:1 poetry run thesis train

# srun -p gpushort --gres=gpu:1 -c 8 --mem 16000 poetry run thesis --threads=8 train --batch-size=32 --no-run-eval --max-epochs=5 --dataset solutions/data/big-gen-toy-100k --num-workers=8

# srun env | sort


# srun poetry run thesis data-gen -c 50 --od-share 0.015 --activity-drop-prob 0.1 \
#     --seed $SLURM_ARRAY_TASK_ID --out solutions/data/grid \
#     --time-limit 120 timpasslib/grid
