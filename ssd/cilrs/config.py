import os
from threading import local

class GlobalConfig:

	# Data
    seq_len = 1 # input timesteps
    pred_len = 0 # future waypoints predicted, not required for CILRS

    local_root_dir = '/home/scholar/tmp/ssd/transfuser/data/processed'
    root_dir = '/home/scholar/tmp/ssd/transfuser/data/14_weathers_minimal_data'
    os.system(f'mkdir -p {local_root_dir}/ssd_data')
    
    train_towns = ['Town01', 'Town02', 'Town03', 'Town04']
    val_towns = ['Town05']
    ssd_towns = [ 'Town06', 'Town07', 'Town10']
    
    train_data, val_data, ssd_data = [], [], []
    for town in train_towns:
        train_data.append(town+'_tiny')
        train_data.append(town+'_short')
    
    for town in ssd_towns:
        ssd_data.append(town+'_tiny')
        ssd_data.append(town+'_short')

    for town in val_towns:
        val_data.append(town+'_short')

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    max_throttle = 0.75 # upper limit on throttle signal value in dataset

    lr = 1e-4 # learning rate

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

        if self.train_with_ssd_data:
            print('Self-supervised Training')
            GlobalConfig.train_data.append(os.path.join(GlobalConfig.local_root_dir, 'ssd_data'))
        else:
            print('Collect data for self-supervised training')
