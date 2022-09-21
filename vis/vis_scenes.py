import matplotlib.pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir
import tqdm
from PIL import Image

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
data = dict(
        turns=[],
        in_motion=[],
        long_stops=[],
        short_stops=[]
    )

for src in tqdm.tqdm(files):
    if ('filtered_data' in src) and ('Town' in src):
        src_dict = np.load(src, allow_pickle=True).item()
        data['turns'].extend(src_dict['turns'])
        data['in_motion'].extend(src_dict['in_motion'])
        data['long_stops'].extend(src_dict['long_stops'])
        data['short_stops'].extend(src_dict['short_stops'])

    # if ('rgb_front' in src) or ('measurements' in src):
    #     dest = dest_root + src[len('/mnt/qb/geiger/kchitta31/datasets/carla/transfuserplus_dataset_21_06/'):]
    #     os.system(f'mkdir -p `dirname {dest}` && cp {src} {dest} &')
for i in tqdm.tqdm(range(len(data['long_stops']))):

    plt.figure()
    # plt.scatter(tp[:,0], tp[:,1])
    # plt.imsave('x.png', Image.open(x['scene']).)
    os.system(f'cp {data["long_stops"][i]["scene"]} x.png')
    os.system(f'cp {data["long_stops"][i+1]["scene"]} y.png')
    a = np.array(Image.open(data["long_stops"][i]["scene"]))
    b = np.array(Image.open(data["long_stops"][i+1]["scene"]))
    print(np.linalg.norm(a-b))
    input()