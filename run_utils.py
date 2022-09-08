import os
import time
import sys
import datetime
import random
import subprocess
import atexit


def get_free_port():
    """
    Returns a free port.
    """

    # get random in between 2000 and 3000 divisble by 5
    port = random.randint(4000, 4500)
    
    #port = 2000
    port_free = False

    while not port_free:
        try:
            pid = int(subprocess.check_output(f"lsof -t -i :{port} -s TCP:LISTEN", shell=True,).decode("utf-8"))
            # print(f'Port {port} is in use by PID {pid}')
            port = random.randint(4000, 4500)

        except subprocess.CalledProcessError:
            port_free = True
            return port
            # print(f'Port {port} is free')


def log(txt):
    print("\n\n\n")
    print("="*len(txt))
    print(txt)
    print("="*len(txt))
    print()
    sys.stdout.flush()


# debugger = f'/usr/bin/env /home/scholar/miniconda3/envs/venv/bin/python /home/scholar/.vscode-server/extensions/ms-python.python-2022.8.0/pythonFiles/lib/python/debugpy/launcher 37701 -- '

# os.system(
#     f'python tools/result_parser.py '+
#     f'--xml /home/scholar/tmp/ssd/transfuser/leaderboard/data/evaluation_routes/routes_town05_long.xml ' + 
#     f'--results /home/scholar/tmp/ssd/saved_results/ '+
#     f'--save_dir /home/scholar/tmp/ssd/saved/'
#     )
# exit()

slurm = '''#!/bin/bash
#SBATCH --job-name=CILRS # Number of tasks (see below)
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=2-00:00            # Runtime in D-HH:MM
#SBATCH --gres=gpu:rtx2080ti:{}    # optionally type and number of gpus
#SBATCH --mem=32G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output={}/%j.out  # File to which STDOUT will be written
#SBATCH --error={}/%j.err   # File to which STDERR will be written
#SBATCH --partition=gpu-2080ti        # gpu-2080ti-preemptable

scontrol show job $SLURM_JOB_ID 
'''

common_exports = '''PYTHONPATH=$PYTHONPATH:{}
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
PORT={}
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
TEAM_AGENT=leaderboard/team_code/{}.py
CHECKPOINT_ENDPOINT=results/{}.json
TEAM_CONFIG={}
SCENARIOS=leaderboard/data/scenarios/town05_all_scenarios.json
'''
leaderboard_evaluator = '''PYTHONPATH=$PYTHONPATH python3 leaderboard/leaderboard/leaderboard_evaluator.py 
--scenarios=${{SCENARIOS}}
--routes=${{ROUTES}}
--repetitions=${{REPETITIONS}}
--track=${{CHALLENGE_TRACK_CODENAME}}
--checkpoint=${{CHECKPOINT_ENDPOINT}}
--agent=${{TEAM_AGENT}}
--agent-config=${{TEAM_CONFIG}}
--debug=${{DEBUG_CHALLENGE}}
--record=${{RECORD_PATH}}
--resume=${{RESUME}}
--port=${{PORT}}
--trafficManagerPort=${{TM_PORT}}
--conf={} &'''
