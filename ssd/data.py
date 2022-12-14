import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset
import imageio
from copy import deepcopy

class CARLA_Data(Dataset):
    def __init__(self, towns, config, len_from_data=False, imgaug=None, use_pseudo_data=False) -> None:
        print("New Dataloader")
        self.config = config
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.imgaug = imgaug
        
        self.data = dict(
            turns=[],
            in_motion=[],
            long_stops=[],
            short_stops=[],
        )

        for town in towns:
            town_data = np.load(f'{self.config["data_dir"]}/{town}/filtered_data.npy', allow_pickle=True).item()
            self.data['turns'].extend(town_data['turns'])
            self.data['in_motion'].extend(town_data['in_motion'])
            self.data['long_stops'].extend(town_data['long_stops'])
            self.data['short_stops'].extend(town_data['short_stops'])
    
        if use_pseudo_data:
            try:
                print("Importing pseudo data")
                ssd_data = np.load(f'{self.config["data_dir"]}/pseudo/{self.config["test_id"]}/filtered_data.npy', allow_pickle=True).item()
                self.data['turns'].extend(ssd_data['turns'])
                self.data['in_motion'].extend(ssd_data['in_motion'])
                self.data['long_stops'].extend(ssd_data['long_stops'])
                self.data['short_stops'].extend(ssd_data['short_stops'])
            except:
                print('There is no pseudolabeled data')


        self.length = (len(self.data['turns'])+len(self.data['in_motion'])+len(self.data['long_stops'])+len(self.data['short_stops'])) if len_from_data else 2000*64

    def __len__(self):
        """Returns the length of the dataset. """
        return self.length

    def __getitem__(self, index):
        try:
            mode = np.random.choice(['turns','in_motion','long_stops', 'short_stops'], p=[0.25, 0.3, 0.25, 0.2])
            index = np.random.randint(0, len(self.data[mode]))
            item = self.data[mode][index]
            img = np.array(scale_and_crop_image(Image.open(item['scene']), scale=self.config['scale'], crop=self.config['input_resolution']))
            
            example = deepcopy(item)
            example['waypoints'] = [tuple(wp) for wp in item['waypoints']]
            example['fronts'] = []
            if self.imgaug is None:
                example['fronts'].append(torch.from_numpy(img.transpose(2,0,1)))
            else:
                aug_img = self.imgaug.augment_image(img)
                # imageio.imwrite(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/scenes/{index}.jpg', aug_img)  #write all changed images
                example['fronts'].append(torch.from_numpy(aug_img.transpose(2,0,1)))
            return example
            
        except:
            return self.__getitem__(index)
        
class CARLA_Data2(Dataset):
    def __init__(self, towns, config, len_from_data=False, imgaug=None, use_pseudo_data=False, what_if=False) -> None:
        print("Old Dataloader")

        self.config = config
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.imgaug = imgaug
        self.what_if = what_if
        self.len_from_data = len_from_data
        
        self.data = []

        for town in towns:
            town_data = list(np.load(f'{self.config["data_dir"]}/{town}/processed_data.npy', allow_pickle=True))
            self.data.extend(town_data)
        
        if use_pseudo_data:
            try:
                print("Importing pseudo data")
                ssd_data = list(np.load(f'{self.config["data_dir"]}/pseudo/{self.config["test_id"]}/processed_data.npy', allow_pickle=True))
                self.data.extend(ssd_data)
            except:
                print('There is no pseudolabeled data')

        self.length = len(self.data) if len_from_data else 2000*64
    
    def __len__(self):
        """Returns the length of the dataset. """
        return self.length

    def __getitem__(self, index):
        if not self.len_from_data:
            index = np.random.randint(0, len(self.data))
        item = self.data[index]
        img = np.array(scale_and_crop_image(Image.open(item['scene']), scale=self.config['scale'], crop=self.config['input_resolution']))
        
        example = deepcopy(item)
        example['waypoints'] = [tuple(wp) for wp in item['waypoints']]
        example['fronts'] = []
        if self.imgaug is None:
            example['fronts'].append(torch.from_numpy(img.transpose(2,0,1)))
        else:
            aug_img = self.imgaug.augment_image(img)
            # imageio.imwrite(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/scenes/{index}.jpg', aug_img)  #write all changed images
            example['fronts'].append(torch.from_numpy(aug_img.transpose(2,0,1)))
        
        if self.what_if:
            example['nav_command'] = np.zeros(6)
            example['nav_command'][np.random.randint(0,4)] = 1

            theta = np.random.random() * np.pi
            example['target_point'] = (50*np.random.random()) * np.array([np.cos(theta), np.sin(theta)])
            example['target_point'][1] = -example['target_point'][1]
            example['target_point'] = tuple(example['target_point'])

        return example



# class CARLA_Data2(Dataset):

#     def __init__(self, root, config, is_imgaug=True):
#         self.config = config
#         self.seq_len = config.seq_len
#         self.pred_len = config.pred_len
#         self.is_imgaug = is_imgaug
#         self.imgaug = iaa.Sequential([
#             iaa.SomeOf((0,2),
#             [
#                 iaa.AdditiveGaussianNoise(scale=0.08*255, per_channel=True),
#                 iaa.AdditiveGaussianNoise(scale=0.08*255),
#                 iaa.Multiply((0.5, 1.5)),
#                 iaa.GaussianBlur(sigma=(0.0, 0.8)),
#                 iaa.Dropout(p=(0, 0.1)),
#                 iaa.SaltAndPepper(0.05),
#             ], random_order=True),
#         ])
#         self.front = []
#         self.left = []
#         self.right = []
#         self.rear = []
#         self.x = []
#         self.y = []
#         self.x_command = []
#         self.y_command = []
#         self.theta = []
#         self.steer = []
#         self.throttle = []
#         self.brake = []
#         self.command = []
#         self.velocity = []
#         self.waypoints = []

#         for sub_root in root:
#             sub_root_local = os.path.join(self.config.local_root_dir, sub_root)
#             sub_root = os.path.join(self.config.root_dir, sub_root)
#             # print(sub_root_local,not os.path.exists(sub_root))
#             if not os.path.exists(sub_root):
#                 continue
#             os.system(f'mkdir -p {sub_root_local}')
#             preload_file = os.path.join(sub_root_local, 'rg_aim_pl_'+str(self.seq_len)+'_'+str(self.pred_len)+'.npy')

#             # dump to npy if no preload
#             if not os.path.exists(preload_file):
#                 preload_front = []
#                 preload_left = []
#                 preload_right = []
#                 preload_rear = []
#                 preload_x = []
#                 preload_y = []
#                 preload_x_command = []
#                 preload_y_command = []
#                 preload_theta = []
#                 preload_steer = []
#                 preload_throttle = []
#                 preload_brake = []
#                 preload_command = []
#                 preload_velocity = []
#                 preload_waypoints = []

#                 # list sub-directories in root 
#                 root_files = os.listdir(sub_root)
#                 routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root,folder))]
#                 for route in routes:
#                     route_dir = os.path.join(sub_root, route)
#                     print(route_dir)
#                     # subtract final frames (pred_len) since there are no future waypoints
#                     # first frame of sequence not used
#                     num_seq = (len(os.listdir(route_dir+"/rgb_front/"))-self.pred_len-2)//self.seq_len
#                     for seq in range(num_seq):
#                         fronts = []
#                         lefts = []
#                         rights = []
#                         rears = []
#                         xs = []
#                         ys = []
#                         thetas = []

#                         # read files sequentially (past and current frames)
#                         for i in range(self.seq_len):
                            
#                             # images
#                             filename = f"{str(seq*self.seq_len+1+i).zfill(4)}.png"
#                             fronts.append(route_dir+"/rgb_front/"+filename)
#                             lefts.append(route_dir+"/rgb_left/"+filename)
#                             rights.append(route_dir+"/rgb_right/"+filename)
#                             rears.append(route_dir+"/rgb_rear/"+filename)

#                             # position
#                             with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
#                                 data = json.load(read_file)
#                             xs.append(data['x'])
#                             ys.append(data['y'])
#                             thetas.append(data['theta'])

#                         # get control value of final frame in sequence
#                         preload_x_command.append(data['x_command'])
#                         preload_y_command.append(data['y_command'])
#                         preload_steer.append(data['steer'])
#                         preload_throttle.append(data['throttle'])
#                         preload_brake.append(data['brake'])
#                         preload_command.append(data['command'])
#                         preload_velocity.append(data['speed'])
#                         preload_waypoints.append([])

#                         # read files sequentially (future frames)
#                         for i in range(self.seq_len, self.seq_len + self.pred_len):
#                             # position
#                             with open(route_dir + f"/measurements/{str(seq*self.seq_len+1+i).zfill(4)}.json", "r") as read_file:
#                                 data = json.load(read_file)
#                             xs.append(data['x'])
#                             ys.append(data['y'])

#                             # fix for theta=nan in some measurements
#                             if np.isnan(data['theta']):
#                                 thetas.append(0)
#                             else:
#                                 thetas.append(data['theta'])

#                         preload_front.append(fronts)
#                         preload_left.append(lefts)
#                         preload_right.append(rights)
#                         preload_rear.append(rears)
#                         preload_x.append(xs)
#                         preload_y.append(ys)
#                         preload_theta.append(thetas)

#                 # dump to npy
#                 preload_dict = {}
#                 preload_dict['front'] = preload_front
#                 preload_dict['left'] = preload_left
#                 preload_dict['right'] = preload_right
#                 preload_dict['rear'] = preload_rear
#                 preload_dict['x'] = preload_x
#                 preload_dict['y'] = preload_y
#                 preload_dict['x_command'] = preload_x_command
#                 preload_dict['y_command'] = preload_y_command
#                 preload_dict['theta'] = preload_theta
#                 preload_dict['steer'] = preload_steer
#                 preload_dict['throttle'] = preload_throttle
#                 preload_dict['brake'] = preload_brake
#                 preload_dict['command'] = preload_command
#                 preload_dict['velocity'] = preload_velocity
#                 preload_dict['waypoints'] = preload_waypoints
#                 np.save(preload_file, preload_dict)

#             # load from npy if available
#             preload_dict = np.load(preload_file, allow_pickle=True)
#             self.front += preload_dict.item()['front']
#             self.x += preload_dict.item()['x']
#             self.y += preload_dict.item()['y']
#             self.theta += preload_dict.item()['theta']
#             self.x_command += preload_dict.item()['x_command']
#             self.y_command += preload_dict.item()['y_command']
#             self.waypoints += preload_dict.item()['waypoints']
#             print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

#     def __len__(self):
#         """Returns the length of the dataset. """
#         return len(self.front)

#     def __getitem__(self, index):
#         """Returns the item at index idx. """
#         data = dict()
#         data['fronts'] = []

#         data['ssd_fronts'] = self.front[index]
#         data['x_command'] = self.x_command[index]
#         data['y_command'] = self.y_command[index]
#         seq_fronts = self.front[index]
#         waypoints = self.waypoints[index]

#         seq_x = self.x[index]
#         seq_y = self.y[index]
#         seq_theta = self.theta[index]

#         for i in range(self.seq_len):
#             img = np.array(scale_and_crop_image(Image.open(seq_fronts[i]), scale=self.config.scale, crop=self.config.input_resolution))
#             if self.is_imgaug:
#                 aug_img = self.imgaug.augment_image(img)
#                 # imageio.imwrite(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/scenes/{index}.jpg', aug_img)  #write all changed images
#                 data['fronts'].append(torch.from_numpy(aug_img.transpose(2,0,1)))
#             else:
#                 data['fronts'].append(torch.from_numpy(img.transpose(2,0,1)))
        
#             # fix for theta=nan in some measurements
#             if np.isnan(seq_theta[i]):
#                 seq_theta[i] = 0.

#         ego_x = seq_x[i]
#         ego_y = seq_y[i]
#         ego_theta = seq_theta[i]   

#         if not waypoints:
#             # lidar and waypoint processing to local coordinates
#             waypoints = []
#             for i in range(self.seq_len + self.pred_len):
#                 # waypoint is the transformed version of the origin in local coordinates
#                 # we use 90-theta instead of theta
#                 # LBC code uses 90+theta, but x is to the right and y is downwards here
#                 local_waypoint = transform_2d_points(np.zeros((1,3)), 
#                     np.pi/2-seq_theta[i], -seq_x[i], -seq_y[i], np.pi/2-ego_theta, -ego_x, -ego_y)
#                 waypoints.append(tuple(local_waypoint[0,:2]))

#         data['waypoints'] = waypoints

#         # convert x_command, y_command to local coordinates
#         # taken from LBC code (uses 90+theta instead of theta)
#         R = np.array([
#             [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
#             [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
#             ])
#         local_command_point = np.array([self.x_command[index]-ego_x, self.y_command[index]-ego_y])
#         local_command_point = R.T.dot(local_command_point)
#         data['target_point'] = tuple(local_command_point)

#         return data


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