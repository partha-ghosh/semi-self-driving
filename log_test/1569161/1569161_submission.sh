#!/bin/bash

# Parameters
#SBATCH --error=/mnt/qb/work/geiger/pghosh58/transfuser/log_test/%j/%j_0_log.err
#SBATCH --job-name=submitit
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/qb/work/geiger/pghosh58/transfuser/log_test/%j/%j_0_log.out
#SBATCH --partition=gpu-2080ti-preemptable
#SBATCH --signal=USR1@90
#SBATCH --time=1
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /mnt/qb/work/geiger/pghosh58/transfuser/log_test/%j/%j_%t_log.out --error /mnt/qb/work/geiger/pghosh58/transfuser/log_test/%j/%j_%t_log.err /mnt/qb/work/geiger/pghosh58/conda/pkgs/transfuser/bin/python -u -m submitit.core._submit /mnt/qb/work/geiger/pghosh58/transfuser/log_test/%j
