#!/bin/bash
#SBATCH --job-name=CILRS # Number of tasks (see below)
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=1-00:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:rtx2080ti:1    # optionally type and number of gpus
#SBATCH --mem=32G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/%j.err   # File to which STDERR will be written
#SBATCH --partition=gpu-2080ti        # gpu-2080ti-preemptable

scontrol show job $SLURM_JOB_ID 
carla_port=`python /mnt/qb/work/geiger/pghosh58/transfuser/tools/get_carla_port.py`
tm_port=$((port+8000))
echo "carla port: $carla_port"
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 /mnt/qb/work/geiger/pghosh58/transfuser/carla/CarlaUE4.sh --world-port=$carla_port -opengl &
sleep 60
cd /mnt/qb/work/geiger/pghosh58/transfuser
PYTHONPATH=$PYTHONPATH:/mnt/qb/work/geiger/pghosh58/transfuser
CARLA_ROOT=carla
CARLA_SERVER=$CARLA_ROOT/CarlaUE4.sh
PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI
PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
PYTHONPATH=$PYTHONPATH:leaderboard
PYTHONPATH=$PYTHONPATH:leaderboard/team_code
PYTHONPATH=$PYTHONPATH:scenario_runner
LEADERBOARD_ROOT=leaderboard
CHALLENGE_TRACK_CODENAME=SENSORS
PORT=$carla_port
TM_PORT=$tm_port
DEBUG_CHALLENGE=0
REPETITIONS=1
ROUTES=leaderboard/data/validation_routes/routes_town05_short.xml
TEAM_AGENT=leaderboard/team_code/auto_pilot.py
TEAM_CONFIG=aim/log/aim_ckpt
CHECKPOINT_ENDPOINT=results/sample_result.json
SCENARIOS=leaderboard/data/scenarios/no_scenarios.json
SAVE_PATH=data/expert
RESUME=False
ROUTES=leaderboard/data/evaluation_routes/routes_town05_long.xml
TEAM_AGENT=leaderboard/team_code/aim_agent.py
CHECKPOINT_ENDPOINT=results/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21_$SLURM_JOB_ID.json
TEAM_CONFIG=/mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21/aim
SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json

PYTHONPATH=$PYTHONPATH python3 leaderboard/leaderboard/leaderboard_evaluator.py  --scenarios=${SCENARIOS} --routes=${ROUTES} --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} --debug=${DEBUG_CHALLENGE} --record=${RECORD_PATH} --resume=${RESUME} --port=${PORT} --trafficManagerPort=${TM_PORT}
sleep 3
mkdir -p /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21/results_$SLURM_JOB_ID
mv /mnt/qb/work/geiger/pghosh58/transfuser/results/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21_$SLURM_JOB_ID.json /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21/results_$SLURM_JOB_ID/result.json
python /mnt/qb/work/geiger/pghosh58/transfuser/tools/result_parser.py --xml /mnt/qb/work/geiger/pghosh58/transfuser/leaderboard/data/evaluation_routes/routes_town05_long.xml --town_maps /mnt/qb/work/geiger/pghosh58/transfuser/leaderboard/data/town_maps_xodr --results /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21/results_$SLURM_JOB_ID --save_dir /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21/results_$SLURM_JOB_ID
pkill -f "port=$carla_port"
mv /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/$SLURM_JOB_ID.out /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21/results_$SLURM_JOB_ID/
mv /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/$SLURM_JOB_ID.err /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21/results_$SLURM_JOB_ID/
python /mnt/qb/work/geiger/pghosh58/transfuser/tools/run_again.py "/mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21/results_$SLURM_JOB_ID/$SLURM_JOB_ID.err" "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21.sh"
mv /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/$SLURM_JOB_ID.out /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21/results_$SLURM_JOB_ID/
mv /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/$SLURM_JOB_ID.err /mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidenece_from_gt/log/aim_confidenece_from_gt:train_n_collect_e60_b64_08_20_03_21/results_$SLURM_JOB_ID/