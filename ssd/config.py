import os

class GlobalConfig:
    """ base architecture configurations """
	# Data
    seq_len = 1 # input timesteps
    pred_len = 4 # future waypoints predicted

    # root_dir = '/mnt/qb/work/geiger/pghosh58/transfuser/data/14_weathers_minimal_data'
    root_dir = '/mnt/qb/work/geiger/pghosh58/transfuser/data/transfuser_plus_data'
    local_root_dir = f'/mnt/qb/work/geiger/pghosh58/transfuser/data/processed/{os.path.basename(root_dir)}'
    
    train_towns = ['Town01', 'Town02', 'Town03', 'Town04']
    val_towns = ['Town05']
    ssd_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town07', 'Town10']
    
    train_data, val_data, ssd_data = [], [], []
    for town in train_towns:
        train_data.append(town+'_tiny')
        train_data.append(town+'_short')
        train_data.append(town+'_long')
    
    for town in ssd_towns:
        ssd_data.append(town+'_tiny')
        ssd_data.append(town+'_short')
        ssd_data.append(town+'_long')

    for town in val_towns:
        val_data.append(town+'_short')

    ignore_sides = True # don't consider side cameras
    ignore_rear = True # don't consider rear cameras

    input_resolution = 256

    scale = 1 # image pre-processing
    crop = 256 # image pre-processing

    lr = 1e-4 # learning rate

    # Controller
    turn_KP = 1.25
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40 # buffer size

    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40 # buffer size

    max_throttle = 0.75 # upper limit on throttle signal value in dataset
    brake_speed = 0.4 # desired speed below which brake is triggered
    brake_ratio = 1.1 # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25 # maximum change in speed input to logitudinal controller

    def __init__(self, **kwargs):
        self.self_supervised_training = False
        for k,v in kwargs.items():
            setattr(self, k, v)
        

        if self.self_supervised_training:
            print('Self-supervised Training')
            GlobalConfig.ssd_train_data = [os.path.join(GlobalConfig.local_root_dir+'/../ssd_data', self.ssd_dir)]
        else:
            print('Supervised Training')

