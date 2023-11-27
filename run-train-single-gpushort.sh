#!/bin/bash -l

#SBATCH --time=0-4
#SBATCH --mem=12G
#SBATCH --job-name=single-run
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mail-type=END,FAIL

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load miniconda
conda activate thesis

srun -p gpushort --gpus=1 -c 4 --mem=12G thesis --threads=4 train \
    --dataset solutions/data/nov-24-preference-fix \
    --batch-size 32 \
    --num-layers 32 \
    --num-heads 4 \
    --num-workers 4 \
    --hidden-channels 32 \
    --lr 0.0003 \
    --accelerator auto \
    --max-epochs 10 \
    --patience 3 \
    --run-name deep-test
