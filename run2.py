from run_utils import *

root = os.path.dirname(os.path.abspath(__file__))
exp_time = datetime.datetime.now().strftime("%m_%d_%H_%M")

tests = [
    # AIM Baseline
    [
        {
            # "name": "aim-14_weathers_minimal_data-supervised",
            "name": "aim-transfuser_plus_data_all_towns_noise_filtered_lr3-supervised",
            "dir": f"{root}/ssd",
            "sst": 0,
            "agent_name": "aim_agent",
            "epochs": 50,
            "batch_size": 64,
            "eval": 3,
            "copy_last_model": 0,
            "load_model": 0,
        },
    #     *[{
    #         # "name": "aim-14_weathers_minimal_data-supervised",
    #         "name": f"aim-transfuser_plus_data-self_supervised_{i}",
    #         "dir": f"{root}/ssd",
    #         "sst": 1,
    #         "agent_name": "aim_agent",
    #         "epochs": 5,
    #         "batch_size": 64,
    #         "eval": 0,
    #         "copy_last_model": 1,
    #         "load_model": 1,
    #     } for i in range(0)],
    #     {
    #         # "name": "aim-14_weathers_minimal_data-supervised",
    #         "name": "aim-transfuser_plus_data2-self_supervised",
    #         "dir": f"{root}/ssd",
    #         "sst": 1,
    #         "agent_name": "aim_agent",
    #         "epochs": 50,
    #         "batch_size": 64,
    #         "eval": 3,
    #         "copy_last_model": 1,
    #         "load_model": 1,
    #     }
    ],

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
    #     # "name": "aim-14_weathers_minimal_data_all_town-supervised",
    #     "name": "aim-14_transfuser_plus_data_all_town-supervised",
    #     # "name": "aim-14_pami_data_all_town-supervised",
    #     "dir": f"{root}/aim",
    #     "sst": '',
    #     "agent_name": "aim_agent",
    #     "epochs": 0,
    #     "batch_size": 64,
    #     "eval": 3,
    #     "copy_last_model": 0,
    #     "load_model": 0,
    # }],
]

# os.system('pkill -f carla')
# time.sleep(5)

def get_test_name(test):
    # test_name = "07_30_10_59-aim-baseline-supervised_e60_b64"
    return exp_time+'-'+test["name"]+f'_e{test["epochs"]}'+f'_b{test["batch_size"]}'

def run_test(tests):
    for test in tests:
        test_name = get_test_name(test)
        test_dir = f'{root}/tmp/{test_name}'
        orig_dir = test['dir']
        splits = test_name.split('-')
        os.system(f'mkdir -p {test_dir} && cd {orig_dir} && cp config.py data.py model.py train.py {splits[1]}.py {test_dir}/')


    cmd_trains = []
    cmd_evals = []
    old_test_dir = None
    for i, test in enumerate(tests):
        carla_port = get_free_port()
        tm_port = carla_port + 8000
        
        # test_name = "07_30_10_59-aim-baseline-supervised_e60_b64"
        test_name = get_test_name(test)
        splits = test_name.split('-')

        orig_dir = test['dir']
        test_dir = f'{root}/tmp/{test_name}'

        cmd_trains.append({
            'test_name':test_name,
            'orig_dir':orig_dir,
            'test_dir':test_dir,
            'cmd': 
            [
                # f'mkdir -p {test_dir} && rsync -a {orig_dir}/* {test_dir}/ --exclude=log* --exclude=__pycache__',
                f'rsync -a {old_test_dir}/log {test_dir}/ --exclude=*.err --exclude=*.out --exclude=*tfevents*' if test['copy_last_model'] else "",
                f'cd {test_dir}',
                f'CUDA_VISIBLE_DEVICES=0 python train.py --framework {splits[1]} --logdir log --epochs {test["epochs"]} --batch_size {test["batch_size"]} --load_model {test["load_model"]} {"--sst {} --ssd_dir {}".format(test["sst"], "-".join(splits[:-1])) if type(test["sst"])==int else ""} --id saved_model',
                f'''{f"cp -r /mnt/qb/work/geiger/pghosh58/transfuser/data/processed/ssd_data/{'-'.join(splits[:-1])} log/" if test["sst"]==0 else "echo"}''',

                # 3 evaluations
                *([f'python {root}/tools/sbatch_submitter.py "sbatch {root}/shell_scripts/run_eval_{test_name}.sh"',]*test["eval"]),
                
                f'python {root}/tools/sbatch_submitter.py "sbatch {root}/shell_scripts/run_train_{get_test_name(tests[i+1])}.sh"' if i<len(tests)-1 else "",

                f'mv {root}/tmp/$SLURM_JOB_ID.out {test_dir}/log/',
                f'mv {root}/tmp/$SLURM_JOB_ID.err {test_dir}/log/',
            ]
        })
        old_test_dir = test_dir
        
        cmd_evals.append(
            {
                'test_name':test_name,
                'orig_dir':orig_dir,
                'test_dir':test_dir,
                'cmd':[
                    f'carla_port=`python /mnt/qb/work/geiger/pghosh58/transfuser/tools/get_carla_port.py`',
                    f'tm_port=$((port+8000))',
                    f'echo "carla port: $carla_port"',
                    f'SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 {root}/carla/CarlaUE4.sh --world-port=$carla_port -opengl &',
                    f'sleep 60',
                    f'cd {root}',
                    common_exports.format(root, test['agent_name'], test_name+'_$SLURM_JOB_ID', f'{test_dir}/log/saved_model'),
                    '{}'.format(leaderboard_evaluator.replace("\n", " ")),# f"{test['dir']}/log/{test_name}/eval.txt"),
                    f'sleep 3',
                    f'mkdir -p {test_dir}/log/results_$SLURM_JOB_ID',
                    f'mv {root}/results/{test_name}_$SLURM_JOB_ID.json {test_dir}/log/results_$SLURM_JOB_ID/result.json',
                    f'python {root}/tools/result_parser.py --xml {root}/leaderboard/data/evaluation_routes/routes_town05_long.xml --town_maps {root}/leaderboard/data/town_maps_xodr --results {test_dir}/log/results_$SLURM_JOB_ID --save_dir {test_dir}/log/results_$SLURM_JOB_ID',
                    f'pkill -f "port=$carla_port"',

                    f'mv {root}/tmp/$SLURM_JOB_ID.out {test_dir}/log/results_$SLURM_JOB_ID/',
                    f'mv {root}/tmp/$SLURM_JOB_ID.err {test_dir}/log/results_$SLURM_JOB_ID/',
                    f'python {root}/tools/run_again.py "{test_dir}/log/results_$SLURM_JOB_ID/$SLURM_JOB_ID.err" "sbatch /mnt/qb/work/geiger/pghosh58/transfuser/shell_scripts/run_eval_{test_name}.sh"',

                ]
            }
        )

    for i, cmd_train in enumerate(cmd_trains):
        with open(f'shell_scripts/run_train_{cmd_train["test_name"]}.sh', 'w') as f:
            f.write(slurm.format(1, f'{root}/tmp',f'{root}/tmp')+"\n".join(cmd_train["cmd"]))
        os.system(f'chmod +x shell_scripts/run_train_{cmd_train["test_name"]}.sh')
        if i==0:
            os.system(f'python {root}/tools/sbatch_submitter.py "sbatch shell_scripts/run_train_{cmd_train["test_name"]}.sh"')

    for cmd_eval in cmd_evals:
        with open(f'shell_scripts/run_eval_{cmd_eval["test_name"]}.sh', 'w') as f:
            f.write(slurm.format(1, f'{root}/tmp',f'{root}/tmp')+"\n".join(cmd_eval["cmd"]))
        os.system(f'chmod +x shell_scripts/run_eval_{cmd_eval["test_name"]}.sh')

for test in tests:
    run_test(test)