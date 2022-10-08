from turtle import heading
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import json
from PIL import Image
from torchvision import models
import os
from os import listdir
from os.path import isfile, join, isdir
import sys
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
from imgaug import augmenters as iaa, imgaug
import imageio
import glob

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

# hyper parameters
num_epochs = 10
batch_size = 6
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
        towns = ['Town01', 'Town02', 'Town03', 'Town04'] if mode=='train' else ['Town05']
        self.driving_scenes = []
        for d in data:
            if 'processed' in d:
                if any([(town in d) for town in towns]):
                    print(d)
                    town_data = np.load(d, allow_pickle=True)
                    self.driving_scenes.extend(town_data)

        self.driving_scenes_dict = {k['scene']:True for k in self.driving_scenes}

    def __getitem__(self, index):
        n = np.random.randint(0, len(self.driving_scenes))
        # batch_size_choices = [1, 2, 4, 8, 16, 32, 64]
        # batch_size = np.random.choice(batch_size_choices)
        # n_seqs = 2*max(batch_size_choices)//batch_size        
        n_seqs = 24

        scene_names = [self.driving_scenes[n]['scene']]
        t = int(scene_names[0][-8:-4])
        for i in range(1, n_seqs+10):
            scene_names.append(scene_names[0].replace(str(t).zfill(4), str(t+i).zfill(4)))
        
        for scene_name in scene_names:
            if not self.driving_scenes_dict.get(scene_name, False):
                return self.__getitem__(index)
        else:
            measurements = []
            abs_pos = []
            rel_translations = []
            for scene_name in scene_names:
                with open(scene_name.replace('.png', '.json').replace('rgb_front', 'measurements'), 'r') as f:
                    measurements.append(json.load(f))

            for measurement in measurements:
                abs_pos.append([np.round(measurement['x'],1), np.round(measurement['y'],1)])
            try:
                abs_pos, remove_first_n = perturb_in_direction(np.array(abs_pos))
                abs_pos[n_seqs+1]
            except:
                return self.__getitem__(index)

            scene_names = scene_names[remove_first_n:]
            # print(abs_pos)
            # plt.figure()
            # plt.plot(abs_pos[:,0], abs_pos[:,1], marker='.')
            # plt.savefig('x.png')
            for i in range(2, len(abs_pos)):
                v = get_rel_pos(abs_pos[i-2], abs_pos[i-1], abs_pos[i])
                rel_translations.append([*v, *(v/np.linalg.norm(v))])
            # print(rel_translations)

            # plt.figure()
            # plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # No decimal places
            # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}')) # No decimal places
            
            # v = abs_pos[1]-abs_pos[0]
            # v = v/np.linalg.norm(v)
            # M = np.array([[v[0], v[1]], [-v[1], v[0]]])
            # abs_pos = (M @ (abs_pos-abs_pos[0]).T).T + abs_pos[0]

            # plt.plot(np.round(abs_pos[1:,0],2), np.round(abs_pos[1:,1],2), marker='.',label='gt')
            # print(abs_pos)
            # abs0 = abs_pos[1]
            # pred_abs_pos = [abs0]
            # c0,s0 = 1,0
            # for x,y,c1,s1 in rel_translations:
            #     abs0 = get_abs_pos(abs0, c0, s0, np.array([x,y]))
            #     c0, s0 = c0*c1-s0*s1, c0*s1+s0*c1
            #     pred_abs_pos.append(abs0)
            
            # pred_abs_pos = np.array(pred_abs_pos)
            # plt.plot(np.round(pred_abs_pos[:,0],2),np.round(pred_abs_pos[:,1],2),marker='.',label='pred_before',alpha=0.5,linewidth=2)
            # print(pred_abs_pos)
            
            # plt.legend()
            # plt.savefig('x.png')
            # exit()

            scenes = []
            for scene_name in scene_names[1:]:
                scenes.append(torch.from_numpy(np.array(scale_and_crop_image(Image.open(scene_name), scale=1, crop=256))))

            # for i in range(len(scenes)):
            #     imageio.imwrite(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/scenes/{i}scenet.jpg', scenes[i])  #write all changed images
            # imageio.imwrite(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/scenes/{index}scenetp1.jpg', scene_tp1)  #write all changed images

            # exit()
            for i in range(len(scenes)):
                scenes[i] = scenes[i].permute(2,0,1).float()

            return torch.cat([torch.cat([scenes[i-1], scenes[i]])[None,:,:,:] for i in range(1,n_seqs+1)]), torch.tensor(rel_translations[:n_seqs], dtype=torch.float)

    def __len__(self):
        return len(self.driving_scenes)

def get_abs_pos(abs0, c0, s0, rel_trans):
    r1 = np.linalg.norm(rel_trans)
    if r1:
        c1, s1 = rel_trans/r1
        
        c0, s0 = c1*c0-s1*s0, c0*s1+s0*c1

    return abs0 + r1 * np.array([c0, s0])

def get_rel_pos(abs0, abs1, abs2):
    tm1_2_t0 = np.array([abs1[0]-abs0[0], abs1[1]-abs0[1]])
    unit_tm1_2_t0 = tm1_2_t0/np.linalg.norm(tm1_2_t0)
    t0_2_tp1 = np.array([abs2[0]-abs1[0], abs2[1]-abs1[1]])
    x_proj = unit_tm1_2_t0.dot(t0_2_tp1)
    x = unit_tm1_2_t0 * x_proj
    y = t0_2_tp1-x
    y_proj = np.linalg.norm(y)
    if np.cross(x, y) < 0:
        y_proj = -y_proj
    return np.array([x_proj, y_proj])


def perturb_in_direction(vec):
    diff = vec[1:] - vec[:-1]
    i = 0

    while np.linalg.norm(diff[i]) == 0:
        vec = np.delete(vec, 1, 0)
        i+=1
    remove_first_n = i
    diff = diff[remove_first_n:]

    for i in range(1,len(diff)):
        if np.linalg.norm(diff[i]) == 0:
            diff[i] = diff[i-1]

    diff = diff/np.linalg.norm(diff).reshape((-1,1))

    i=0
    for j in range(len(diff)):
        if (diff[j]==diff[i]).all():
            diff[j] += diff[j-1]
        else:
            i = j

    vec[1:] += 0.00001*diff

    return vec, remove_first_n



# CIFAR10
dataset_path  = '/mnt/qb/work/geiger/pghosh58/transfuser/data/filtered_new'
train_dataset = MyDataset(dataset_path, 'train')
test_dataset = MyDataset(dataset_path, 'test')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class RelNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoding = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1), #256
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), #128
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), #64
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(256, 512, 3, padding=1), #16
            nn.ReLU(),
            nn.MaxPool2d(4,4),
            nn.Conv2d(512, 1024, 3, padding=1), #4
            nn.ReLU(),
            nn.MaxPool2d(4,4),                  #1
        )
        self.fc1 = nn.Linear(1024,256)

        self.decoder = nn.GRUCell(input_size=256, hidden_size=256)
        self.regressor = nn.Linear(256, 4)

    def forward(self, seqs):
        pred_rel_translations = []
        for seq in seqs:
            x = self.encoding(seq)
            x = x.view(-1, 1024)
            x = torch.relu(self.fc1(x))

            z = torch.zeros_like(x[0])
            tmp = []
            for i in range(len(seq)):
                z = self.decoder(x[i], z)
                dx = self.regressor(z)
                tmp.append(dx[None,:])
            pred_rel_translations.append(torch.cat(tmp)[None, :,:])
        return torch.cat(pred_rel_translations)

model = RelNet()
# model = nn.DataParallel(model)
model.to(device)

try:
  model.load_state_dict(torch.load('model.pth'))
  model.eval()
except:
  pass

# loss and optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
old_loss = 9999999
train_iter = 0
test_iter = 0

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (scenes, rel_motions) in enumerate(train_loader):
        scenes = scenes.to(device)
        rel_motions = rel_motions.to(device)

        #forward pass
        pred_rel_motions = model(scenes)
        loss = criterion(10*pred_rel_motions, 10*rel_motions)
        
        writer.add_scalar('train_loss', loss.item(), train_iter)
        train_iter += 1
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i)%50 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
            torch.save(model.state_dict(), f'model.pth')
            sys.stdout.flush()
    # test
            with torch.no_grad():
                cum_loss = 0
                for i, (scenes, rel_motions) in enumerate(test_loader):
                    scenes = scenes.to(device)
                    rel_motions = rel_motions.to(device)

                    #forward pass
                    pred_rel_motions = model(scenes)
                    loss = criterion(pred_rel_motions, rel_motions)

                    cum_loss += loss

                    if i == 10:
                        break
                

                rel_motions = rel_motions.cpu().numpy()
                pred_rel_motions = pred_rel_motions.cpu().numpy()
                for k in range(len(rel_motions[0])):
                    plt.figure()
                    plt.scatter([rel_motions[0][k][0]], [rel_motions[0][k][1]], marker='.',label='gt_p')
                    plt.scatter([pred_rel_motions[0][k][0]], [pred_rel_motions[0][k][1]],marker='.',label='pred_p')
                    plt.scatter([rel_motions[0][k][2]], [rel_motions[0][k][3]], marker='*',label='gt_d')
                    plt.scatter([pred_rel_motions[0][k][2]], [pred_rel_motions[0][k][3]],marker='*',label='pred_d')
                    plt.legend()
                    plt.savefig(f'img/{k}.png')
                    
                loss = cum_loss / (i+1)
                writer.add_scalar('test_loss', loss.item(), test_iter)
                test_iter += 1
                if loss < old_loss:
                    torch.save(model.state_dict(), f'best_model.pth')
                    old_loss = loss
                print(f'epoch {epoch+1} / {num_epochs}, test loss = {loss.item():.4f}')
