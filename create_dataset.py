from cmath import isnan
import os
from os import listdir
from os.path import isfile, join, isdir
import cv2
import os
from scipy.ndimage.measurements import label
import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from copy import deepcopy
import argparse
import multiprocessing as mp


def getAllFilesRecursive(root):
    files = [ join(root,f) for f in listdir(root) if isfile(join(root,f))]
    dirs = [ d for d in listdir(root) if isdir(join(root,d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root,d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root,f))
    return files

def get_scenes(town_path):
    files = getAllFilesRecursive(town_path)
    scenes = []
    for f in files:
        if '.png' in f:
            if 'rgb_front' in f:
                scenes.append(f)

    return scenes

def get_measurements(scene):
    with open(scene.replace('png','json').replace('rgb_front','measurements'), 'r') as f:
        data = json.load(f)
    return data


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:,2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T
    
    # reset z-coordinate
    out[:,2] = xyz[:,2]

    return out

def calc_wp(seq_x, seq_y, seq_theta):

    i = seq_len-1
    ego_x = seq_x[i]
    ego_y = seq_y[i]
    ego_theta = seq_theta[i]
    
    waypoints = []
    for i in range(seq_len + pred_len):
        # waypoint is the transformed version of the origin in local coordinates
        # we use 90-theta instead of theta
        # LBC code uses 90+theta, but x is to the right and y is downwards here
        local_waypoint = transform_2d_points(np.zeros((1,3)), 
            np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta, -ego_x, -ego_y)
        waypoints.append(tuple(local_waypoint[0,:2]))
    return waypoints


# x = []
# y = []
# z = []
# scenes = get_scenes('/mnt/qb/work/geiger/pghosh58/transfuser/data/14_weathers_minimal_data/Town06_long/routes_town06_11_05_23_28_28')
# for scene in tqdm.tqdm(scenes):
#     index = int(scene[-8:-4])
#     measurements = get_measurements(scene)
#     y.append(measurements['steer'])
#     # x.append(measurements['throttle'])
#     z.append(measurements['speed'])
#     # y.append(1 if measurements['brake'] else 0)

# z = np.array(z)
# # plt.plot(z,label='speed')
# plt.plot(y,label='steer')
# r=np.minimum(1/(np.abs(z)+0.01),10)
# r = r[1:]-r[:-1]
# p = np.maximum(savgol_filter(np.abs(r), 11, 3),0)
# # plt.plot(p)
# s = np.where(p>0.2,1,0)
# # plt.plot(s, label='stop')
# q = (1-s)
# q = q*(z[1:]>1)
# # plt.plot(q, label='move')
# # plt.legend()
# # plt.savefig('x.png')
# # exit()


data_path = '/mnt/qb/work/geiger/pghosh58/transfuser/data/14_weathers_minimal_data'
# data_path = '/mnt/qb/work/geiger/pghosh58/transfuser/data/transfuser_plus_data'
dest_path = '/'.join([('filtered_'+x if '_data' in x else x) for x in data_path.split('/')])
print(dest_path)
seq_len = 1
pred_len = 4
towns = next(os.walk(data_path))[1]

def worker(routes, data_path, seq_len, pred_len, dest_path):
    data = dict()
    for route in tqdm.tqdm(routes, desc='route', leave=False):
        scenes = get_scenes(f'{data_path}/{town}/{route}')
        for scene in tqdm.tqdm(scenes, desc='scene', leave=False):
            try:
                index = int(scene[-8:-4])
                measurements = get_measurements(scene)
                
                ego_x = measurements['x']
                ego_y = measurements['y']
                ego_theta = np.pi if np.isnan(measurements['theta']) else measurements['theta']
                x_command = measurements['x_command']
                y_command = measurements['y_command']

                nav_command = np.zeros(6)
                nav_command[measurements['command']-1] = 1

                R = np.array([
                    [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
                    [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
                    ])
                local_command_point = np.array([x_command-ego_x, y_command-ego_y])
                local_command_point = R.T.dot(local_command_point)

                seq_x, seq_y, seq_theta = [], [], []
                c_img = str(index).zfill(4)
                for i in range(index, index+seq_len+pred_len):    
                    f_img = str(i).zfill(4)
                    measurements = get_measurements(scene.replace(c_img, f_img))
                    seq_x.append(measurements['x'])
                    seq_y.append(measurements['y'])
                    seq_theta.append(np.pi if np.isnan(measurements['theta']) else measurements['theta'])
                waypoints = calc_wp(seq_x, seq_y, seq_theta)
                
                v1 = np.linalg.norm(data.get(scene.replace(c_img, str(index-1).zfill(4)), {'waypoints':[0,0]})['waypoints'][1])                
                v2 = np.linalg.norm(data.get(scene.replace(c_img, str(index-2).zfill(4)), {'waypoints':[0,0]})['waypoints'][1])
                data[scene] = dict(
                    v2 = v2,
                    v1 = v1,
                    scene=scene,
                    target_point=tuple(local_command_point),
                    waypoints=waypoints,
                    nav_command=nav_command,
                )
            except:
                pass
            # print(data)
            # input()
    os.system(f'mkdir -p {dest_path}/{town}')
    np.save(f'{dest_path}/{town}/processed_data.npy', list(data.values()))


jobs = []
for town in tqdm.tqdm(towns, desc='town'):
    routes = next(os.walk(f'{data_path}/{town}'))[1]
    # worker(routes, data_path, seq_len, pred_len, dest_path)
    p = mp.Process(target=worker, args=(routes, data_path, seq_len, pred_len, dest_path))
    jobs.append(p)
    p.start()
for p in jobs:
    p.join()


for town in tqdm.tqdm(towns, desc='town'):
    data = dict(
        turns=[],
        in_motion=[],
        stops=[]
    )
    town_path = f'{dest_path}/{town}/processed_data.npy'
    processed_data = np.load(town_path, allow_pickle=True)
    routes = dict()

    for item in processed_data:
        route = '/'.join(item['scene'].split('/')[:-2])
        routes.setdefault(route, [])
        routes[route].append(item)

    for route in tqdm.tqdm(routes, desc='route', leave=False):
        seqs = routes[route]
        speed_list = []
        theta_list = []
        for item in tqdm.tqdm(seqs, desc='scene', leave=False):
            speed_list.append(np.linalg.norm(item['waypoints'][1]))
            theta = np.arctan2(item['waypoints'][-1][1],item['waypoints'][-1][0])+(np.pi/2)
            theta_list.append(0 if abs(theta)>1 else abs(theta))
        speed_list = 2*np.array(speed_list)
        theta_list = np.array(theta_list)
        # print(speed_list)
        
        # filtered_theta = savgol_filter(theta_list, 21, 3, mode='nearest')
        steer_indicator = np.where(theta_list>0.07,1,0)

        move_indicator = np.where(speed_list>2 , 1, 0)
        stop_indicator = 1 - move_indicator

        move_arg_indicator = np.argwhere(move_indicator).flatten()
        m = min(move_arg_indicator) if len(move_arg_indicator) else 0
        M = max(move_arg_indicator) if len(move_arg_indicator) else len(stop_indicator)
        stop_indicator[:m] = 0
        stop_indicator[M:] = 0

        window_size = 3
        stop_indicator_copy = deepcopy(stop_indicator)
        for i in range(window_size, len(stop_indicator)-window_size-1):
            stop_indicator[i] = (stop_indicator[i] if (stop_indicator_copy[i-window_size:i+window_size+1]==0).any() else 0)

        steer_indicator = steer_indicator * move_indicator
        window_size = 1
        steer_indicator_copy = deepcopy(steer_indicator)
        for i in range(window_size, len(steer_indicator)-window_size-1):
            steer_indicator[i] = (1 if (steer_indicator_copy[i-window_size:i+window_size+1]==1).any() else steer_indicator[i])

        # plt.figure()
        # plt.plot(speed_list,label='speed from wp')
        # plt.plot(theta_list,label='theta from wp')
        # plt.plot(stop_indicator*0.75, label='stop wp')
        # # plt.plot(filtered_theta,label='filtered theta from wp')
        # plt.plot(steer_indicator*0.5,label='steer indicator from wp')
        # plt.plot(move_indicator, label='move wp')
        # plt.title(f'{route[-50:]}')
        # plt.legend()
        # plt.savefig('x.png')
        # input()

        for i in range(len(move_indicator)):
            if move_indicator[i]:
                data['in_motion'].append(seqs[i])
            if stop_indicator[i]:
                data['stops'].append(seqs[i])
            if steer_indicator[i]:
                data['turns'].append(seqs[i])
        # print(data)
        # input()

    np.save(f'{dest_path}/{town}/filtered_data.npy', data)
