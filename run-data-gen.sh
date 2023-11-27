#!/bin/bash -l

#SBATCH --time=00-20
#SBATCH --mem=2G
#SBATCH --job-name=data-gen
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=END,FAIL
#SBATCH --array=1-40
#SBATCH --output=slurm/main/%A_%a.log
#SBATCH --error=slurm/main/%A_%a.log


export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load miniconda
conda activate thesis


srun -c 2 --mem=2G thesis --threads=2 multi-solution-generation \
    --seed $SLURM_ARRAY_TASK_ID --out solutions/data/nov-24-preference-fix \
    --time-limit 20 --count=400 --alt-solutions=10

