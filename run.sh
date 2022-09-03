#!/bin/bash
#SBATCH --job-name=CILRS # Number of tasks (see below)
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=2-00:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:rtx2080ti:1    # optionally type and number of gpus
#SBATCH --mem=32G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/geiger/pghosh58/transfuser/%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/geiger/pghosh58/transfuser/%j.err   # File to which STDERR will be written
#SBATCH --partition=gpu-2080ti        # gpu-2080ti-preemptable

scontrol show job $SLURM_JOB_ID 
cd /mnt/qb/work/geiger/pghosh58/transfuser
CUDA_VISIBLE_DEVICES=0 python img2video.py
mv $SLURM_JOB_ID.out out.txt
mv $SLURM_JOB_ID.err err.txt