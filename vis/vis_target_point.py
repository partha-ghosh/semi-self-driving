import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir
import tqdm

def getAllFilesRecursive(root):
    files = [ join(root,f) for f in listdir(root) if isfile(join(root,f))]
    dirs = [ d for d in listdir(root) if isdir(join(root,d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root,d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root,f))
    return files

files = getAllFilesRecursive('/mnt/qb/work/geiger/pghosh58/transfuser/data/filtered_14_weathers_minimal_data')
data = []

for src in tqdm.tqdm(files):
    if ('processed' in src) and ('Town' in src):
        data.extend(np.load(src, allow_pickle=True))

    # if ('rgb_front' in src) or ('measurements' in src):
    #     dest = dest_root + src[len('/mnt/qb/geiger/kchitta31/datasets/carla/transfuserplus_dataset_21_06/'):]
    #     os.system(f'mkdir -p `dirname {dest}` && cp {src} {dest} &')
tp = []
for d in data:
    tp.append(np.array(d['target_point']))
tp = np.array(tp)

plt.figure()
plt.scatter(tp[:,0], tp[:,1])
plt.savefig('x.png')