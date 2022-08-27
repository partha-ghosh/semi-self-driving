from run_utils import *

root = os.path.dirname(os.path.abspath(__file__))
exp_time = datetime.datetime.now().strftime("%m_%d_%H_%M")

tests = [
    # AIM Baseline
    # [
    #     {
    #         "name": "train_n_collect",
    #         "dir": f"{root}/ssd/aim",
    #         "sst": 0,
    #         "agent_name": "aim_agent",
    #         "epochs": 1,
    #         "batch_size": 64,
    #     },
    #     {
    #         "name": "self_supervised",
    #         "dir": f"{root}/ssd/aim",
    #         "sst": 1,
    #         "agent_name": "aim_agent",
    #         "epochs": 1,
    #         "batch_size": 64,
    #     }
    # ],

    # AIM Noise
    # [
    #     {
    #         "name": "aim_noise:supervised_training",
    #         "dir": f"{root}/ssd/aim_noise",
    #         "sst": 0,
    #         "agent_name": "aim_agent",
    #         "epochs": 60,
    #         "batch_size": 64,
    #     },
    #     {
    #         "name": "aim_noise:self_supervised_training_1",
    #         "dir": f"{root}/ssd/aim_noise",
    #         "sst": 1,
    #         "agent_name": "aim_agent",
    #         "epochs": 60,
    #         "batch_size": 64,
    #     },
    #     {
    #         "name": "aim_noise:self_supervised_training_2",
    #         "dir": f"{root}/ssd/aim_noise",
    #         "sst": 1,
    #         "agent_name": "aim_agent",
    #         "epochs": 60,
    #         "batch_size": 64,
    #     }
    # ],

    # AIM No Noise
    # [
    #     {
    #         "name": "aim_no_noise:supervised_training",
    #         "dir": f"{root}/ssd/aim_no_noise",
    #         "sst": 0,
    #         "agent_name": "aim_agent",
    #         "epochs": 60,
    #         "batch_size": 64,
    #     },
    #     {
    #         "name": "aim_no_noise:self_supervised_training_1",
    #         "dir": f"{root}/ssd/aim_no_noise",
    #         "sst": 1,
    #         "agent_name": "aim_agent",
    #         "epochs": 60,
    #         "batch_size": 64,
    #     },
    #     {
    #         "name": "aim_no_noise:self_supervised_training_2",
    #         "dir": f"{root}/ssd/aim_no_noise",
    #         "sst": 1,
    #         "agent_name": "aim_agent",
    #         "epochs": 60,
    #         "batch_size": 64,
    #     }
    # ],

    # AIM Confidence
    # [
    #     {
    #         "name": "aim_confidenece:train_n_collect",
    #         "dir": f"{root}/ssd/aim_confidence",
    #         "sst": 0,
    #         "agent_name": "aim_confidence_agent",
    #         "epochs": 60,
    #         "batch_size": 64,
    #     },
    #     {
    #         "name": "aim_confidenece:self_supervised",
    #         "dir": f"{root}/ssd/aim",
    #         "sst": 1,
    #         "agent_name": "aim_agent",
    #         "epochs": 100,
    #         "batch_size": 64,
    #     }
    # ],

    # AIM Confidence from GT
    # [
    #     {
    #         "name": "aim_confidence_from_gt:train_n_collect",
    #         "dir": f"{root}/ssd/aim_confidence_from_gt",
    #         "sst": 0,
    #         "agent_name": "aim_agent",
    #         "epochs": 60,
    #         "batch_size": 64,
    #     },
    #     {
    #         "name": "aim_confidence_from_gt:self_supervised",
    #         "dir": f"{root}/ssd/aim_confidence_from_gt",
    #         "sst": 1,
    #         "agent_name": "aim_agent",
    #         "epochs": 100,
    #         "batch_size": 64,
    #     }
    # ],
    
    # AIM VO
    # [
    #     {
    #         "name": "aim_vo",
    #         "dir": f"{root}/ssd/aim_vo",
    #         "sst": '',
    #         "agent_name": "aim_agent",
    #         "epochs": 60,
    #         "batch_size": 64,
    #     }
    # ],
    
    # {
    #     "name": "modified_aim_all_town_e50_b128",
    #     "dir": f"{root}/ssd/aim",
    #     "sst": 0,
    #     "agent_name":"ssd_aim_agent",
    #     "epochs": 50,
    #     "batch_size": 128,
    # },
    # [{
    #     "name": "dino_all_town_e50_b128",
    #     "dir": f"{root}/aim_dino",
    #     "sst": '',
    #     "agent_name": "dino_aim_agent",
    #     "epochs": 50,
    #     "batch_size": 64,
    # }],
    
    # [{
    #     "name": "aim_all_town",
    #     "dir": f"{root}/aim",
    #     "sst": '',
    #     "agent_name": "aim_agent",
    #     "epochs": 60,
    #     "batch_size": 64,
    # }],
]

# os.system('pkill -f carla')
# time.sleep(5)

def run_test(tests):
    cmd_train = []
    cmd_evals = []
    for test in tests:
        carla_port = get_free_port()
        tm_port = carla_port + 8000

        test_name = test["name"]+f'_e{test["epochs"]}'+f'_b{test["batch_size"]}_'+exp_time
        cmd_train.extend([

            f'cd {test["dir"]}',
            f'CUDA_VISIBLE_DEVICES=0 python train.py --logdir log/{test_name} --epochs {test["epochs"]} {"--sst {}".format(test["sst"]) if test["sst"] else ""} --batch_size {test["batch_size"]}',
            f'{f"cp -r /mnt/qb/work/geiger/pghosh58/transfuser/data/processed/ssd_data log/{test_name}/" if not test["sst"] else "echo"}',

            # 3 evaluations
            f'python {root}/tools/sbatch_submitter.py "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_{test_name}.sh"',
            f'python {root}/tools/sbatch_submitter.py "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_{test_name}.sh"',
            f'python {root}/tools/sbatch_submitter.py "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_{test_name}.sh"',

            f'mv {test["dir"]}/log/$SLURM_JOB_ID.out {test["dir"]}/log/{test_name}/',
            f'mv {test["dir"]}/log/$SLURM_JOB_ID.err {test["dir"]}/log/{test_name}/',
        ])
        
        cmd_evals.append(
            {
                'test_name':test_name,
                'test_dir':test["dir"],
                'cmd':[
                    f'carla_port=`python /mnt/qb/work/geiger/pghosh58/transfuser/tools/get_carla_port.py`',
                    f'tm_port=$((port+8000))',
                    f'echo "carla port: $carla_port"',
                    f'SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 {root}/carla/CarlaUE4.sh --world-port=$carla_port -opengl &',
                    f'sleep 60',
                    f'cd {root}',
                    common_exports.format(root, test['agent_name'], test_name+'_$SLURM_JOB_ID', f'{test["dir"]}/log/{test_name}/aim'),
                    '{}'.format(leaderboard_evaluator.replace("\n", " ")),# f"{test['dir']}/log/{test_name}/eval.txt"),
                    f'sleep 3',
                    f'mkdir -p {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID',
                    f'mv {root}/results/{test_name}_$SLURM_JOB_ID.json {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID/result.json',
                    f'python {root}/tools/result_parser.py --xml {root}/leaderboard/data/evaluation_routes/routes_town05_long.xml --town_maps {root}/leaderboard/data/town_maps_xodr --results {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID --save_dir {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID',
                    f'pkill -f "port=$carla_port"',

                    f'mv {test["dir"]}/log/$SLURM_JOB_ID.out {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID/',
                    f'mv {test["dir"]}/log/$SLURM_JOB_ID.err {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID/',
                    f'python {root}/tools/run_again.py "{test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID/$SLURM_JOB_ID.err" "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_{test_name}.sh"',

                    f'mv {test["dir"]}/log/$SLURM_JOB_ID.out {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID/',
                    f'mv {test["dir"]}/log/$SLURM_JOB_ID.err {test["dir"]}/log/{test_name}/results_$SLURM_JOB_ID/',
                ]
            }
        )

    cur_time = time.time()
    with open(f'shell_scripts/run{cur_time}.sh', 'w') as f:
        f.write(slurm.format(1, f'{test["dir"]}/log',f'{test["dir"]}/log')+"\n".join(cmd_train))
    os.system(f"chmod +x shell_scripts/run{cur_time}.sh")
    os.system(f'python {root}/tools/sbatch_submitter.py "sbatch shell_scripts/run{cur_time}.sh"')

    for cmd_eval in cmd_evals:
        with open(f'shell_scripts/run_{cmd_eval["test_name"]}.sh', 'w') as f:
            f.write(slurm.format(1, f'{cmd_eval["test_dir"]}/log',f'{cmd_eval["test_dir"]}/log')+"\n".join(cmd_eval["cmd"]))
        os.system(f'chmod +x shell_scripts/run_{cmd_eval["test_name"]}.sh')

for test in tests:
    run_test(test)