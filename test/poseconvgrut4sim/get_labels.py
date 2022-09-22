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
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True
from imgaug import augmenters as iaa
import imageio
import matplotlib.pyplot as plt

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

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

imgaug = iaa.Sequential([
    	iaa.SomeOf((0, 1),
    	[
			# iaa.AdditiveGaussianNoise(scale=0.2*255, per_channel=True),
		    # iaa.Dropout(p=(0, 0.2)),
			# iaa.SaltAndPepper(0.1),
		], random_order=True),
    ])

class MyDataset(Dataset):

    def __init__(self, mode, imgaug):
        # data loading
        if mode == 'train':
            if os.path.exists('/mnt/qb/work/geiger/pghosh58/transfuser/data/train_driving_scenes.npy'):
                self.driving_scenes = np.load('/mnt/qb/work/geiger/pghosh58/transfuser/data/train_driving_scenes.npy',allow_pickle='TRUE')
                self.driving_scenes.sort()
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
        n = index
        # batch_size_choices = [1, 2, 4, 8, 16, 32, 64]
        # batch_size = np.random.choice(batch_size_choices)
        # n_seqs = 2*max(batch_size_choices)//batch_size        
        n_seqs = 3

        scene_names = [self.driving_scenes[n]] 
        t = int(scene_names[0][-8:-4])

        for i in range(1, n_seqs):
            scene_names.append(scene_names[0].replace(str(t).zfill(4), str(t+i).zfill(4)))
        
        for scene_name in scene_names:
            if not self.driving_scenes_dict.get(scene_name, False):
                return None,None,None
        else:
            measurements = []
            rel_translations = []
            for scene_name in scene_names:
                with open(scene_name.replace('.png', '.json').replace('rgb_front', 'measurements'), 'r') as f:
                    measurements.append(json.load(f))

            for i in range(2, len(measurements)):
                tm1_2_t0 = np.array([measurements[i-2]['x']-measurements[i-1]['x'], measurements[i-2]['y']-measurements[i-1]['y']])
                unit_tm1_2_t0 = tm1_2_t0/np.linalg.norm(tm1_2_t0)
                t0_2_tp1 = np.array([measurements[i-1]['x']-measurements[i]['x'], measurements[i-1]['y']-measurements[i]['y']])
                x_proj = unit_tm1_2_t0.dot(t0_2_tp1)
                x = unit_tm1_2_t0 * x_proj
                y = t0_2_tp1-x
                y_proj = np.linalg.norm(y)
                if np.cross(x, y) > 0:
                    y_proj = -y_proj
                if np.isnan(x_proj):
                    x_proj = 0.0
                if np.isnan(y_proj):
                    y_proj = 0.0
                rel_translations.append((x_proj, y_proj))

            scenes = []
            for scene_name in scene_names[1:]:
                scenes.append(torch.from_numpy(self.imgaug.augment_image(np.array(scale_and_crop_image(Image.open(scene_name), scale=1, crop=256)))))

            # imageio.imwrite(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/scenes/{index}scenet.jpg', scene_t)  #write all changed images
            # imageio.imwrite(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/scenes/{index}scenetp1.jpg', scene_tp1)  #write all changed images
            
            for i in range(len(scenes)):
                scenes[i] = scenes[i].permute(2,0,1).float()

            return torch.cat([torch.cat([scenes[i-1], scenes[i]])[None,:,:,:] for i in range(1,len(scenes))]), torch.tensor(rel_translations, dtype=torch.float), t==0

    def __len__(self):
        return len(self.driving_scenes)


# CIFAR10
train_dataset = MyDataset('train', imgaug)
test_dataset = MyDataset('test', imgaug)

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
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
        self.decoder = nn.GRUCell(input_size=256, hidden_size=256)
        self.regressor = nn.Linear(256, 2)

    # def forward(self, seqs):
    #     pred_rel_translations = []
    #     for seq in seqs:
    #         x = self.encoding(seq)
    #         x = x.view(-1, 1024)
    #         x = torch.relu(self.fc1(x))
    #         z = torch.zeros_like(x[0])
    #         tmp = []
    #         for i in range(len(seq)):
    #             z = self.decoder(x[i], z)
    #             dx = self.output(z)
    #             tmp.append(dx[None,:])
    #         pred_rel_translations.append(torch.cat(tmp)[None, :,:])
    #     return torch.cat(pred_rel_translations)
    
    def forward(self, seq, z):
        x = self.encoding(seq)
        x = x.view(-1, 1024)
        x = torch.relu(self.fc1(x))
        
        z = self.decoder(x, z)
        dx = self.regressor(z)
        return dx, z

model = RelNet().to(device)

try:
  model.load_state_dict(torch.load('best_model.pth'))
  model.eval()
  print('model loaded')
except:
  pass

rel_x = []
rel_y = []
with torch.no_grad():
    for i in range(len(train_dataset.driving_scenes)):
        (scenes, rel_motions, is_start) = test_dataset[i]
        # if scenes is not None:
        #     rel_x.append(rel_motions[0][0])
        #     rel_y.append(rel_motions[0][1])
        # if len(rel_x) == 5000:
        #     break

        if is_start:
            z = torch.zeros((1,256)).to(device)
        scenes = scenes.to(device)

        #forward pass
        pred_rel_motions, z = model(scenes, z)
        print(pred_rel_motions)
        print(rel_motions)

plt.scatter(rel_x, rel_y, alpha=0.3)
plt.savefig('x.png')
        
#     loss = cum_loss / (i+1)
#     writer.add_scalar('test_loss', loss.item(), test_iter)
#     test_iter += 1
#     if loss < old_loss:
#         torch.save(model.state_dict(), f'best_model.pth')
#         old_loss = loss
#     print(f'epoch {epoch+1} / {num_epochs}, test loss = {loss.item():.4f}')



# # loss and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# old_loss = 9999999
# train_iter = 0
# test_iter = 0

# # training loop
# n_total_steps = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (scenes, rel_motions) in enumerate(train_loader):
#         scenes = scenes.to(device)
#         rel_motions = rel_motions.to(device)

#         #forward pass
#         pred_rel_motions = model(scenes)
#         loss = criterion(pred_rel_motions, rel_motions)
        
#         writer.add_scalar('train_loss', loss.item(), train_iter)
#         train_iter += 1
#         # backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i+1)%50 == 0:
#             print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
#             torch.save(model.state_dict(), f'model.pth')
#             sys.stdout.flush()
#     # test
#             with torch.no_grad():
#                 cum_loss = 0
#                 for i, (scenes, rel_motions) in enumerate(test_loader):
#                     scenes = scenes.to(device)
#                     rel_motions = rel_motions.to(device)

#                     #forward pass
#                     pred_rel_motions = model(scenes)
#                     loss = criterion(pred_rel_motions, rel_motions)

#                     cum_loss += loss

#                     if i == 4:
#                         break
                    
#                 loss = cum_loss / (i+1)
#                 writer.add_scalar('test_loss', loss.item(), test_iter)
#                 test_iter += 1
#                 if loss < old_loss:
#                     torch.save(model.state_dict(), f'best_model.pth')
#                     old_loss = loss
#                 print(f'epoch {epoch+1} / {num_epochs}, test loss = {loss.item():.4f}')
