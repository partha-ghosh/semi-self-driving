import sys
from pprint import pprint
config = dict()
exec(f'config = {sys.argv[1]}')
pprint(config)

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

from imgaug import augmenters as iaa, imgaug

from model import AIM

from trainer import Trainer, step
# if config['use_nav']:
# 	AIM.train = train_with_nav
# if config['use_acc']:
# 	AIM.train = train_with_acc
# else:
Trainer.step = step


from data import CARLA_Data2
if config['dataloader'] == 1:
    from data import CARLA_Data
elif config['dataloader'] == 0:
    from data import CARLA_Data2 as CARLA_Data




if config['imgaug']:
    imgaug = iaa.Sequential([
                iaa.SomeOf((0,2),
                [
                    iaa.AdditiveGaussianNoise(scale=0.08*255, per_channel=True),
                    iaa.AdditiveGaussianNoise(scale=0.08*255),
                    iaa.Multiply((0.5, 1.5)),
                    iaa.GaussianBlur(sigma=(0.0, 0.8)),
                    iaa.Dropout(p=(0, 0.1)),
                    iaa.SaltAndPepper(0.05),
                ], random_order=True),
            ])
else:
    imgaug = None




config['logdir'] = os.path.join(config['logdir'], 'saved_model')

writer = SummaryWriter(log_dir=config['logdir'])

# Data
# train_set = CARLA_Data(root=config.train_data, config=config)
ssd_set = CARLA_Data2(towns=config['self_supervised_towns'], config=config, imgaug=None, len_from_data=True)
val_set = CARLA_Data(towns=config['validation_towns'], config=config, imgaug=None, len_from_data=True)

# dataloader_train = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
dataloader_ssd = DataLoader(ssd_set, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
dataloader_val = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

# Model
model = AIM(config, config['device'])
optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
trainer = Trainer(config=config, writer=writer, model=model, optimizer=optimizer, val_dataloader=dataloader_val, ss_dataloader=dataloader_ssd)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print ('Total trainable parameters: ', params)

if config['load_model']:
    try:
        model.load_state_dict(torch.load(os.path.join(config['logdir'], 'model.pth')))
        print("model loaded")
    except:
        raise 'failed to load the model'

if config['training_type'][:2] == 'ss':
    if config['training_type'] == 'ssf':
        print("Training with Pseudolabels")
        train_set = CARLA_Data(towns=[], config=config, imgaug=imgaug, use_pseudo_data=True)
        n_epochs = config['epochs']//2
    elif config['training_type'] == 'ssgt':
        print("Training with Pseudolabels and GT")
        train_set = CARLA_Data(towns=config['supervised_towns'], config=config, imgaug=imgaug, use_pseudo_data=True)
        n_epochs = config['epochs']

    dataloader_train = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    trainer.train_dataloader = dataloader_train

    for epoch in range(trainer.cur_epoch, n_epochs): 
        trainer.train()
        if epoch % config['val_every'] == 0: 
            trainer.validate()
            trainer.save()
    print("Collect Labels")
    trainer.get_labels()

if config['training_type'] == 'ssf':
    print("Fine Tuning")
    train_set = CARLA_Data(towns=config['supervised_towns'], config=config)
    dataloader_train = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    trainer.train_dataloader = dataloader_train
   
    for epoch in range(trainer.cur_epoch, config['epochs']): 
        trainer.train()
        if epoch % config['val_every'] == 0: 
            trainer.validate()
            trainer.save()

if config['training_type'] == 's':
    print("Supervised Training")
    train_set = CARLA_Data(towns=config['supervised_towns'], config=config, imgaug=imgaug, len_from_data=False)
    dataloader_train = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    trainer.train_dataloader = dataloader_train
    
    for epoch in range(trainer.cur_epoch, config['epochs']): 
        trainer.train()
        if epoch % config['val_every'] == 0: 
            trainer.validate()
            trainer.save()
    print("Collect Labels")
    trainer.get_labels()
