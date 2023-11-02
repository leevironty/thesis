PHONY: pull
pull:
	rsync -av triton:thesis/solutions .

PHONY: logs
logs:
	rsync -av triton:thesis/slurm .

PHONY: push
push:
	rsync -av --exclude-from=.gitignore ../thesis triton:/scratch/work/rontyl1

PHONY: run
run: push
	ssh triton 'cd /scratch/work/rontyl1/thesis && sbatch --mail-user=${AALTO_MAIL} run.sh'

# PHONY: test
# test:
# 	ssh triton "echo $$SHELL sbatch --mail-user=${AALTO_MAIL} run.sh"

status:
	ssh triton "slurm q"

PHONY: snapshot
snapshot:
	tar -cvzf snapshot-$(date +%Y-%m-%d-%H:%M).tar.gz solutions
