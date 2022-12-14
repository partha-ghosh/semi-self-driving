import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque

import torch
import carla
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from ssd.model import AIM
from aim.config import GlobalConfig
from aim.data import scale_and_crop_image
from team_code.planner import RoutePlanner
from wand.image import Image as WandImage
import matplotlib.pyplot as plt
import time

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
    return 'AIMAgent'



class AIMAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.input_buffer = {'rgb': deque(), 'rgb2': deque(), 'rgb_left': deque(), 'rgb_right': deque(), 
                            'rgb_rear': deque(), 'gps': deque(), 'thetas': deque()}


        class Dict2Class(object):
      
            def __init__(self, my_dict):
                
                for key in my_dict:
                    setattr(self, key, my_dict[key])
        
        # self.config = GlobalConfig()
        exec(f'self.conf = {self.conf}')
        self.config = Dict2Class(self.conf)
        self.net = AIM(self.conf, 'cuda')
        self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'model.pth')))
        self.net.cuda()
        self.net.eval()

        self.v1 = 0.2
        self.v2 = 0.2

        self.save_path = None
        # if SAVE_PATH is not None:
        #     now = datetime.datetime.now()
        #     string = pathlib.Path(os.environ['ROUTES']).stem + '_'
        #     string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

        #     print (string)

        #     self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
        #     self.save_path.mkdir(parents=True, exist_ok=False)

        #     (self.save_path / 'rgb').mkdir()
        #     (self.save_path / 'meta').mkdir()

        now = datetime.datetime.now()
        self.save_path2 = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
        self.save_path2 = f'{self.config.test_dir}/scenes/{self.save_path2}'
        print(self.save_path2)
        os.system(f'mkdir -p {self.save_path2}/rgb')
        os.system(f'mkdir -p {self.save_path2}/meta')

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z':2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z':2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 960, 'height': 480, 'fov': 120,
                    'id': 'rgb2'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z':2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb_left'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z':2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb_right'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -1.3, 'y': 0.0, 'z':2.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
                    'width': 400, 'height': 300, 'fov': 100,
                    'id': 'rgb_rear'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]

    def tick(self, input_data):
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        
        rgb2 = cv2.cvtColor(input_data['rgb2'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        img = WandImage.from_array(rgb2)
        img.virtual_pixel = 'transparent'
        img.distort('barrel', (0.0, 0, 0, 1.85))
        img2 = np.array(img)
        center = img2.shape
        w = 400
        h = 300
        x = center[1]/2 - w/2
        y = center[0]/2 - h/2
        rgb2 = img2[int(y):int(y+h), int(x):int(x+w)][:, :, :3]
        
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        result = {
                'rgb': rgb,
                'rgb2': rgb2,
                'rgb_left': rgb_left,
                'rgb_right': rgb_right,
                'rgb_rear': rgb_rear,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                }
        
        pos = self._get_position(result)
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_command'] = next_cmd.value

        theta = compass + np.pi/2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])

        local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        return result


    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        if self.step < self.config.seq_len:
            rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
            
            rgb2 = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb2']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb2'].append(rgb2.to('cuda', dtype=torch.float32))

            if not self.config.ignore_sides:
                rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
                self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
                
                rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
                self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))

            if not self.config.ignore_rear:
                rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
                self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            
            return control

        gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
        command = torch.FloatTensor([tick_data['next_command']]).to('cuda', dtype=torch.float32)

        tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
                                            torch.FloatTensor([tick_data['target_point'][1]])]
        target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

        encoding = []
        scene_type = self.config.scene_type # rgb or rgb2
        rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data[scene_type]), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
        self.input_buffer[scene_type].popleft()
        self.input_buffer[scene_type].append(rgb.to('cuda', dtype=torch.float32))
        encoding.append(self.net.image_encoder(list(self.input_buffer[scene_type])))
        
        if not self.config.ignore_sides:
            rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_left'].popleft()
            self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
            encoding.append(self.net.image_encoder(list(self.input_buffer['rgb_left'])))
            
            rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_right'].popleft()
            self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))
            encoding.append(self.net.image_encoder(list(self.input_buffer['rgb_right'])))

        if not self.config.ignore_rear:
            rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), scale=self.config.scale, crop=self.config.input_resolution)).unsqueeze(0)
            self.input_buffer['rgb_rear'].popleft()
            self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))
            encoding.append(self.net.image_encoder(list(self.input_buffer['rgb_rear'])))

        v1 = torch.tensor([[self.v1]]).to('cuda', dtype=torch.float32)
        v2 = torch.tensor([[self.v2]]).to('cuda', dtype=torch.float32)
        
        nav_command = np.zeros(6)
        nav_command[int(command.item())-1] = 1
        nav_command = torch.tensor(nav_command[:None]).to('cuda', dtype=torch.float32)

        pred_wp = self.net(feature_emb=encoding, v1=v1, v2=v2, target_point=target_point, nav_command=nav_command)[:,:,:2]

        self.v2 = self.v1
        self.v1 = torch.norm(pred_wp[0][1]).cpu().item()
        
        steer, throttle, brake, metadata = self.net.control_pid(pred_wp, gt_velocity)
        self.pid_metadata = metadata

        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        # if SAVE_PATH is not None and self.step % 10 == 0:
        if self.step % 10 == 0:
            self.save(tick_data)

        return control

    def save(self, tick_data):
        frame = self.step // 10

        Image.fromarray(tick_data[self.config.scene_type]).save(f'{self.save_path2}/rgb/{str(frame).zfill(4)}.png')

        outfile = open(f'{self.save_path2}/meta/{str(frame).zfill(4)}.json', 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()

    def destroy(self):
        del self.net
