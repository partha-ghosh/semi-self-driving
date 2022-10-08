#!/bin/bash
#SBATCH --job-name=CILRS # Number of tasks (see below)
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=2-00:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:rtx2080ti:1    # optionally type and number of gpus
#SBATCH --mem=32G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/geiger/pghosh58/transfuser/tmp/%j.out  # File to which STDOUT will be written
#SBATCH --error=/mnt/qb/work/geiger/pghosh58/transfuser/tmp/%j.err   # File to which STDERR will be written
#SBATCH --partition=gpu-2080ti        # gpu-2080ti-preemptable

scontrol show job $SLURM_JOB_ID 
carla_port=`python /mnt/qb/work/geiger/pghosh58/transfuser/tools/get_carla_port.py`
tm_port=$((carla_port+8000))
echo "carla port: $carla_port"
SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 /mnt/qb/work/geiger/pghosh58/transfuser/carla/CarlaUE4.sh --world-port=$carla_port -opengl &
sleep 60
cd /mnt/qb/work/geiger/pghosh58/transfuser


export PYTHONPATH=$PYTHONPATH:'/home/partha/Documents/notes/academics/university/sem4/thesis/code/transfuser'
export CARLA_ROOT=carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
PORT=$carla_port
TM_PORT=$tm_port
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
# export ROUTES=leaderboard/data/validation_routes/routes_town05_short.xml
# export TEAM_AGENT=leaderboard/team_code/auto_pilot.py # agent
# export TEAM_CONFIG=aim/log/aim_ckpt # model checkpoint, not required for expert
# export CHECKPOINT_ENDPOINT=results/sample_result.json # results file
# export SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
# export SAVE_PATH=data/expert # path for saving episodes while evaluating
export RESUME=False


#export ROUTES=leaderboard/data/evaluation_routes/routes_town05_long.xml
#export SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
export ROUTES=/mnt/qb/work/geiger/pghosh58/transfuser/leaderboard/data/training_routes/routes_town10_short.xml
export TEAM_AGENT=leaderboard/team_code/auto_pilot.py
export TEAM_CONFIG=model_ckpt/aim
export CHECKPOINT_ENDPOINT=results/aim_result$SLURM_JOB_ID.json
export SCENARIOS=/mnt/qb/work/geiger/pghosh58/transfuser/leaderboard/data/scenarios/town10_all_scenarios.json
export SAVE_PATH=data/new2/Town10_short # path for saving episodes while evaluating


PYTHONPATH=$PYTHONPATH python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py --scenarios=${SCENARIOS}  --routes=${ROUTES} --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} --debug=${DEBUG_CHALLENGE} --record=${RECORD_PATH} --resume=${RESUME} --port=${PORT} --trafficManagerPort=${TM_PORT} --conf=""

