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
import multiprocessing as mp
import sys
import glob

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

def calc_wp(seq_x, seq_y, seq_theta, seq_len, pred_len):

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

def filter_data(path_to_npy, data):
    processed_data = np.load(path_to_npy, allow_pickle=True)
    routes = dict()

    for item in processed_data:
        item['waypoints']=[tuple(x) for x in item['waypoints']]
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

        window_size = 2
        short_stop_indicator = np.zeros_like(stop_indicator)
        long_stop_indicator = np.zeros_like(stop_indicator)
        for i in range(window_size, len(stop_indicator)-window_size-1):
            window1 = stop_indicator[i-window_size:i+1]
            window2 = stop_indicator[i:i+window_size+1]
            if ((window1==0).any() and (window1==1).any()) or ((window2==0).any() and (window2==1).any()):
                short_stop_indicator[i] = 1
            elif ((window1==1).all() and (window2==1).all()):
                long_stop_indicator[i] = 1
        
        steer_indicator = steer_indicator * move_indicator
        window_size = 1
        steer_indicator_copy = deepcopy(steer_indicator)
        for i in range(window_size, len(steer_indicator)-window_size-1):
            steer_indicator[i] = (1 if (steer_indicator_copy[i-window_size:i+window_size+1]==1).any() else steer_indicator[i])

        # plt.figure()
        # plt.plot(speed_list,label='speed from wp')
        # plt.plot(theta_list,label='theta from wp')
        # plt.plot(stop_indicator*0.75, label='stop wp')
        # plt.plot(long_stop_indicator*0.66, label='long stop wp')
        # plt.plot(short_stop_indicator*1.25, label='short stop wp')
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
            if long_stop_indicator[i]:
                data['long_stops'].append(seqs[i])
            if short_stop_indicator[i]:
                data['short_stops'].append(seqs[i])
            if steer_indicator[i]:
                data['turns'].append(seqs[i])
        # print(data)
        # input()


def worker(town, routes, data_path, seq_len, pred_len, dest_path):
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
                waypoints = calc_wp(seq_x, seq_y, seq_theta, seq_len, pred_len)
                
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

