import os
from subprocess import Popen, PIPE
import sys
import time

flag = True
cmd = 'squeue | grep ghosh' 
while flag:
    time.sleep(10)
    p = Popen(cmd, shell=True, stdout=PIPE, text=True)
    n_sbatch = p.stdout.read().count('\n')
    if n_sbatch < 100:
        flag = False
os.system(f'{sys.argv[1]}')
