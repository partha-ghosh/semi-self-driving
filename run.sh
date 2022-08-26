#!/bin/bash
#SBATCH --job-name=CILRS # Number of tasks (see below)
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=1-00:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:rtx2080ti:1    # optionally type and number of gpus
#SBATCH --mem=32G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt/log/%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt/log/%j.err   # File to which STDERR will be written
#SBATCH --partition=gpu-2080ti        # gpu-2080ti-preemptable

scontrol show job $SLURM_JOB_ID 
cd /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt
CUDA_VISIBLE_DEVICES=0 python train.py --logdir log/aim_confidence_from_gt:train_n_collect_e60_b64_08_20_03_21 --epochs 60  --batch_size 64
cp -r /mnt/qb/work/geiger/pghosh58/transfuser/data/processed/ssd_data log/aim_confidence_from_gt:train_n_collect_e60_b64_08_20_03_21/
python /mnt/qb/work/geiger/pghosh58/transfuser/tools/sbatch_submitter.py "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_aim_confidence_from_gt:train_n_collect_e60_b64_08_20_03_21.sh"
python /mnt/qb/work/geiger/pghosh58/transfuser/tools/sbatch_submitter.py "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_aim_confidence_from_gt:train_n_collect_e60_b64_08_20_03_21.sh"
python /mnt/qb/work/geiger/pghosh58/transfuser/tools/sbatch_submitter.py "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_aim_confidence_from_gt:train_n_collect_e60_b64_08_20_03_21.sh"
mv /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt/log/$SLURM_JOB_ID.out /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt/log/aim_confidence_from_gt:train_n_collect_e60_b64_08_20_03_21/
mv /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt/log/$SLURM_JOB_ID.err /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt/log/aim_confidence_from_gt:train_n_collect_e60_b64_08_20_03_21/
cd /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt
CUDA_VISIBLE_DEVICES=0 python train.py --logdir log/aim_confidence_from_gt:self_supervised_e100_b64_08_20_03_21 --epochs 100 --sst 1 --batch_size 64
echo
python /mnt/qb/work/geiger/pghosh58/transfuser/tools/sbatch_submitter.py "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_aim_confidence_from_gt:self_supervised_e100_b64_08_20_03_21.sh"
python /mnt/qb/work/geiger/pghosh58/transfuser/tools/sbatch_submitter.py "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_aim_confidence_from_gt:self_supervised_e100_b64_08_20_03_21.sh"
python /mnt/qb/work/geiger/pghosh58/transfuser/tools/sbatch_submitter.py "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_aim_confidence_from_gt:self_supervised_e100_b64_08_20_03_21.sh"
mv /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt/log/$SLURM_JOB_ID.out /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt/log/aim_confidence_from_gt:self_supervised_e100_b64_08_20_03_21/
mv /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt/log/$SLURM_JOB_ID.err /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt/log/aim_confidence_from_gt:self_supervised_e100_b64_08_20_03_21/