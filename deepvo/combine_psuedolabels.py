from cProfile import label
from turtle import forward
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from torchvision import models
import os
from os import listdir
from os.path import isfile, join, isdir
import sys
# from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
from imgaug import augmenters as iaa
import imageio
import matplotlib.pyplot as plt
import tqdm
import sys
import glob
import time
from utils import filter_data

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# writer = SummaryWriter()

# hyper parameters
num_epochs = 10
batch_size = 1
learning_rate = 2e-5

# dataset has PILImage images of range [0, 1].
# we transform them to Tensors of normalized range [-1, 1]


def scale_and_crop_image(image, scale=1, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height//2 - crop//2
    start_y = width//2 - crop//2
    cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    # cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class MyDataset(Dataset):

    def __init__(self, datset_path, mode):
        # data loading
        data = glob.glob(f'{dataset_path}/**/*')
        towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town07', 'Town10']
        self.driving_scenes = []
        for d in data:
            if 'processed' in d:
                if any([(town in d) for town in towns]):
                    print(d)
                    town_data = np.load(d, allow_pickle=True)
                    self.driving_scenes.extend(town_data)

        self.driving_scenes_dict = {k['scene']:True for k in self.driving_scenes}

    def __getitem__(self, index):
        n = index
        # batch_size_choices = [1, 2, 4, 8, 16, 32, 64]
        # batch_size = np.random.choice(batch_size_choices)
        # n_seqs = 2*max(batch_size_choices)//batch_size        
        n_seqs = 30

        scene_names = [self.driving_scenes[n]['scene']] 
        t = int(scene_names[0][-8:-4])
        
        for i in range(1, n_seqs):
            scene_names.append(scene_names[0].replace(str(t).zfill(4), str(t+i).zfill(4)))
        
        for scene_name in scene_names:
            if not self.driving_scenes_dict.get(scene_name, False):
                return None,None,None
        else:
            measurements = []
            abs_pos = []
            for scene_name in scene_names:
                with open(scene_name.replace('.png', '.json').replace('rgb_front', 'measurements'), 'r') as f:
                    measurements.append(json.load(f))
        
            for measurement in measurements:
                abs_pos.append((measurement['x'], measurement['y']))

            scenes = []
            for scene_name in scene_names:
                scenes.append(torch.from_numpy(np.array(scale_and_crop_image(Image.open(scene_name), scale=1, crop=256))))

            # imageio.imwrite(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/scenes/{index}.png', scenes[-1])  #write all changed images
            # imageio.imwrite(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/scenes/{index}scenetp1.jpg', scene_tp1)  #write all changed images
            
            for i in range(len(scenes)):
                scenes[i] = scenes[i].permute(2,0,1).float()

            return torch.cat([torch.cat([scenes[i-1], scenes[i]])[None,:,:,:] for i in range(1,len(scenes))]), abs_pos, scene_names[0]

    def __len__(self):
        return len(self.driving_scenes)

def get_abs_pos(abs0, c0, s0, rel_trans):
    r1 = np.linalg.norm(rel_trans)
    c1, s1 = rel_trans/r1
    
    c0, s0 = c1*c0-s1*s0, c0*s1+s0*c1

    return abs0 + r1 * np.array([c0, s0])

# CIFAR10
dataset_path  = '/mnt/qb/work/geiger/pghosh58/transfuser/data/filtered_14_weathers_minimal_data'
# dataset_path  = '/mnt/qb/work/geiger/pghosh58/transfuser/data/filtered_transfuser_plus_data'
train_dataset = MyDataset(dataset_path, 'train')
n = len(train_dataset)
print(f'dataset contains {n} demonstrations')

# i = 0
# while i < (n//10000):
#     print(i)
#     os.system(f'python /mnt/qb/work/geiger/pghosh58/transfuser/run_generic.py 1 "gen_pseudolabels.py {10000*i} 10000"')
#     time.sleep(2)
#     i += 1


i=1
a = np.load('psuedo_waypoints0.npy', allow_pickle=True).item()
while i < n//10000:
    print(i)
    b = np.load(f'psuedo_waypoints{10000*i}.npy', allow_pickle=True).item()
    a.update(b)
    i += 1
np.save(f'processed_data.npy', list(a.values()))
data = dict(
    turns=[],
    in_motion=[],
    long_stops=[],
    short_stops=[]
)
town_path = f'processed_data.npy'
filter_data(town_path, data)
np.save(f'filtered_data.npy', data)
