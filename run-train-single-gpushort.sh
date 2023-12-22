#!/bin/bash -l

#SBATCH --time=1-0
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
    --batch-size 16 \
    --num-layers 16 \
    --num-heads 16 \
    --num-workers 4 \
    --hidden-channels 352 \
    --lr 0.001 \
    --accelerator auto \
    --max-epochs 40 \
    --patience 40 \
    --run-name deep-test-wide-dropout-clips-grad-acc
