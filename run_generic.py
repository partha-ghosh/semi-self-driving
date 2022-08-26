from run_utils import *
import sys
import os

# root = '/mnt/qb/work/geiger/pghosh58/transfuser/test/poseconvgrut4l1'
# root = '/mnt/qb/work/geiger/pghosh58/transfuser/test/poseconvgru_diectt4l1'
root = os.getcwd()

cmd = [
    f'cd {root}',
    f'CUDA_VISIBLE_DEVICES={",".join([str(x) for x in range(int(sys.argv[1]))])} python {sys.argv[2]}',
    f'mv $SLURM_JOB_ID.out out.txt',
    f'mv $SLURM_JOB_ID.err err.txt',
]

with open(f'run.sh', 'w') as f:
    f.write(slurm.format(sys.argv[1], f'{root}',f'{root}')+"\n".join(cmd))
os.system("chmod +x run.sh")
os.system(f'sbatch run.sh')


# i=0
# step = 10000
# while i < 150000:
    

#     cmd = [
#         f'cd {root}',
#         f'CUDA_VISIBLE_DEVICES={",".join([str(x) for x in range(int(sys.argv[1]))])} python {sys.argv[2]} {i} {step}',
#         f'mv $SLURM_JOB_ID.out out.txt',
#         f'mv $SLURM_JOB_ID.err err.txt',
#     ]

#     i+=step

#     with open(f'run.sh', 'w') as f:
#         f.write(slurm.format(sys.argv[1], f'{root}',f'{root}')+"\n".join(cmd))
#     os.system("chmod +x run.sh")
#     os.system(f'sbatch run.sh')