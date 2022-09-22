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
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
from imgaug import augmenters as iaa
import imageio

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

# hyper parameters
num_epochs = 10
batch_size = 32
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

imgaug = iaa.Sequential([
    	iaa.SomeOf((0,2),
    	[
			iaa.Add((-40, 40)),
			iaa.AdditiveGaussianNoise(scale=0.2*255, per_channel=True),
		    iaa.GaussianBlur(sigma=(0.0, 1.0)),
			iaa.Dropout(p=(0, 0.2)),
			iaa.SaltAndPepper(0.1),
		], random_order=True),
		iaa.SomeOf((0,1),
		[	
			iaa.Clouds(),
			iaa.Fog(),
			iaa.Snowflakes(),
			iaa.Rain()
		], random_order=True)
    ])

class MyDataset(Dataset):

    def __init__(self, mode, imgaug):
        # data loading
        if mode == 'train':
            if os.path.exists('/mnt/qb/work/geiger/pghosh58/transfuser/data/train_driving_scenes.npy'):
                self.driving_scenes = np.load('/mnt/qb/work/geiger/pghosh58/transfuser/data/train_driving_scenes.npy',allow_pickle='TRUE')
            else:
                driving_scenes = self.getAllFilesRecursive('/mnt/qb/work/geiger/pghosh58/transfuser/data/14_weathers_minimal_data')
                self.driving_scenes = [x for x in driving_scenes if ('.png' in x and 'Town05' not in x)]
                np.save('/mnt/qb/work/geiger/pghosh58/transfuser/data/train_driving_scenes.npy', self.driving_scenes)
        if mode == 'test':
            if os.path.exists('/mnt/qb/work/geiger/pghosh58/transfuser/data/test_driving_scenes.npy'):
                self.driving_scenes = np.load('/mnt/qb/work/geiger/pghosh58/transfuser/data/test_driving_scenes.npy',allow_pickle='TRUE')
            else:
                driving_scenes = self.getAllFilesRecursive('/mnt/qb/work/geiger/pghosh58/transfuser/data/14_weathers_minimal_data')
                self.driving_scenes = [x for x in driving_scenes if ('.png' in x and 'Town05' in x)]
                np.save('/mnt/qb/work/geiger/pghosh58/transfuser/data/test_driving_scenes.npy', self.driving_scenes)
        
        self.driving_scenes_dict = {k:True for k in self.driving_scenes}
        self.imgaug = imgaug

    def getAllFilesRecursive(self, root):
        files = [ join(root,f) for f in listdir(root) if isfile(join(root,f))]
        dirs = [ d for d in listdir(root) if isdir(join(root,d))]
        for d in dirs:
            files_in_d = self.getAllFilesRecursive(join(root,d))
            if files_in_d:
                for f in files_in_d:
                    files.append(join(root,f))
        return files


    def __getitem__(self, index):
        n = np.random.randint(0, len(self.driving_scenes))
        
        scene_t = self.driving_scenes[n]
        t = int(scene_t[-8:-4])
        scene_tp1 = scene_t.replace(str(t), str(t+1))

        if not self.driving_scenes_dict.get(scene_tp1, False):
            return self.__getitem__(index)
        else:
            with open(scene_t.replace('.png', '.json').replace('rgb_front', 'measurements'), 'r') as f:
                measurement_t = json.load(f)
            with open(scene_tp1.replace('.png', '.json').replace('rgb_front', 'measurements'), 'r') as f:
                measurement_tp1 = json.load(f)
            
            x_rel = measurement_tp1['x']-measurement_t['x']
            y_rel = measurement_tp1['y']-measurement_t['y']

            scene_t = torch.from_numpy(self.imgaug.augment_image(np.array(scale_and_crop_image(Image.open(scene_t), scale=1, crop=256))))
            scene_tp1 = torch.from_numpy(self.imgaug.augment_image(np.array(scale_and_crop_image(Image.open(scene_tp1), scale=1, crop=256))))

            # imageio.imwrite(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/scenes/{index}scenet.jpg', scene_t)  #write all changed images
            # imageio.imwrite(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/scenes/{index}scenetp1.jpg', scene_tp1)  #write all changed images
            
            scene_t = scene_t.permute(2,0,1)
            scene_tp1 = scene_tp1.permute(2,0,1)

            return torch.cat((scene_t.float(), scene_tp1.float())), torch.tensor([x_rel, y_rel], dtype=torch.float)*10

    def __len__(self):
        return len(self.driving_scenes)


# CIFAR10
train_dataset = MyDataset('train', imgaug)
test_dataset = MyDataset('test', imgaug)

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
            nn.MaxPool2d(4,4),
        )
        self.fc1 = nn.Linear(1024,256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64,2)

    def forward(self,x):
        x = self.encoding(x)
        x = x.view(-1, 1024)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = RelNet().to(device)

try:
  model.load_state_dict(torch.load('model.pth'))
  model.eval()
except:
  pass

# loss and optimizer
criterion = nn.MSELoss()
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
        loss = criterion(pred_rel_motions, rel_motions)
        
        writer.add_scalar('train_loss', loss.item(), train_iter)
        train_iter += 1
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%50 == 0:
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

                    if i == 4:
                        break
                    
                loss = cum_loss / (i+1)
                writer.add_scalar('test_loss', loss.item(), test_iter)
                test_iter += 1
                if loss < old_loss:
                    torch.save(model.state_dict(), f'best_model.pth')
                    old_loss = loss
                print(f'epoch {epoch+1} / {num_epochs}, test loss = {loss.item():.4f}')
