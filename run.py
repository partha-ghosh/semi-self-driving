from run_utils import *

root = '/mnt/qb/work/geiger/pghosh58/transfuser'
exp_time = '07_02_01_50' #datetime.datetime.now().strftime("%m_%d_%H_%M")

tests = [
    # [
        # {
        #     "name": "train_n_collect",
        #     "desc": "SSD-CILRS: Training with the new data and Evaluating and Collecting psuedo labels",
        #     "dir": f"{root}/ssd/aim",
        #     "sst": 0,
        #     "agent_name": "aim_agent",
        #     "epochs": 50,
        #     "batch_size": 64,
        # },
    #     {
    #         "name": "self_supervised",
    #         "desc": "SSD-CILRS: Self-supervised training and Evaluating",
    #         "dir": f"{root}/ssd/aim",
    #         "sst": 1,
    #         "agent_name": "aim_agent",
    #         "epochs": 50,
    #         "batch_size": 64,
    #     }
    # ],
    # [{
    #     "name": "modified_aim_all_town_e50_b128",
    #     "desc": "Train on all towns",
    #     "dir": f"{root}/ssd/aim",
    #     "sst": 0,
    #     "agent_name":"aim_agent",
    #     "epochs": 50,
    #     "batch_size": 128,
    # }],
    # [{
    #     "name": "aim_nav_all_town_e50_b128",
    #     "desc": "aim_nav Train on all towns",
    #     "dir": f"{root}/ssd/aim_nav",
    #     "sst": 0,
    #     "agent_name":"ssd_aim_nav_agent",
    #     "epochs": 100,
    #     "batch_size": 128,
    # }],
    # [{
    #     "name": "dino_all_town_e50_b128",
    #     "desc": "Training Baseline AIM with DINO",
    #     "dir": f"{root}/aim_dino",
    #     "sst": '',
    #     "agent_name": "dino_aim_agent",
    #     "epochs": 50,
    #     "batch_size": 64,
    # }],
    [{
        "name": "aim_all_town",
        "desc": "Training Baseline AIM",
        "dir": f"{root}/aim2",
        "sst": '',
        "agent_name": "aim_agent2",
        "epochs": 50,
        "batch_size": 128,
    }],
]

# os.system('pkill -f carla')
# time.sleep(5)

def run_test(tests):
    cmd = []
    for test in tests:

        test_name = 'aim_train_n_collect_06_27_23_15'# test["name"]+f'_e{test["epochs"]}'+f'_b{test["batch_size"]}_'+exp_time
        cmd.extend([
            f'echo "{test_name}"',

            # f'cd {test["dir"]}',
            # f'CUDA_VISIBLE_DEVICES=0 python train.py --logdir log/{test_name} --epochs {test["epochs"]} {"--sst {}".format(test["sst"]) if test["sst"] else ""} --batch_size {test["batch_size"]}',

            f'carla_port=`python /mnt/qb/work/geiger/pghosh58/transfuser/tools/get_carla_port.py`',
            f'tm_port=$((port+8000))',
            f'echo "carla port: $carla_port"',
            f'SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 {root}/carla/CarlaUE4.sh --world-port=$carla_port -opengl &',
            f'sleep 120',
            f'cd {root}',
            common_exports.format(root, test['agent_name'], test_name+'_$SLURM_JOB_ID', f'{test["dir"]}/log/{test_name}/aim'),
            '{}'.format(leaderboard_evaluator.replace("\n", " ")),# f"{test['dir']}/log/{test_name}/eval.txt"),
            f'sleep 3',
            f'mkdir -p {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID',
            f'mv {root}/results/{test_name}_$SLURM_JOB_ID.json {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID/result.json',
            f'python {root}/tools/result_parser.py --xml {root}/leaderboard/data/evaluation_routes/routes_town05_long.xml --town_maps {root}/leaderboard/data/town_maps_xodr --results {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID --save_dir {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID',
            f'pkill -f "port=$carla_port"',

            f'mv {test["dir"]}/log/$SLURM_JOB_ID.out {test["dir"]}/log/{test_name}/',
            f'mv {test["dir"]}/log/$SLURM_JOB_ID.err {test["dir"]}/log/{test_name}/',
        ])

    with open(f'run.sh', 'w') as f:
        f.write(slurm.format(1, f'{test["dir"]}/log',f'{test["dir"]}/log')+"\n".join(cmd))
    os.system("chmod +x run.sh")
    os.system(f'sbatch run.sh')

for test in tests:
    run_test(test)