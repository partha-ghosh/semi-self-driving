import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
import os

os.system("rm -rf /home/scholar/tmp/ssd/transfuser/vis/wp/*")


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

def calc_wp(seq_x, seq_y, seq_theta):

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

# pseudo_labels = np.load('/mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence/log/aim_confidenece:train_n_collect_e60_b64_08_20_03_21/ssd_data/rg_aim_pl_1_4.npy', allow_pickle=True)
pseudo_labels = np.load('/mnt/qb/work/geiger/pghosh58/transfuser/ssd/aim_confidence_from_gt/log/aim_confidence_from_gt:train_n_collect_e60_b64_08_20_03_21/ssd_data/rg_aim_pl_1_4.npy', allow_pickle=True)
pseudo_labels_vo = np.load('/mnt/qb/work/geiger/pghosh58/transfuser/test/poseconvgrut4l1/psuedo_waypoints.npy', allow_pickle=True)


def find_ri():
    n = len(pseudo_labels.item()['front'])
    flag = True
    while flag:
        ri = np.random.randint(0, n)
        flag = True if (
            ('town01' in pseudo_labels.item()['front'][ri][0]) or \
            ('town02' in pseudo_labels.item()['front'][ri][0]) or \
            ('town03' in pseudo_labels.item()['front'][ri][0]) or \
            ('town04' in pseudo_labels.item()['front'][ri][0])
            ) else False
    return ri

ri = find_ri()

M = np.array([[0, 1], [-1, 0]])

index = None
save_index = 0
while True:
    if index is not None:
        try:
            scene = scene.replace(str(index).zfill(4), str(index+1).zfill(4))
            ri = pseudo_labels.item()['front'].index([scene])
        except:
            # exit()
            ri = find_ri()


    scene = pseudo_labels.item()['front'][ri][0]
    index = int(scene[-8:-4])

    print(scene)

    pred_wp = pseudo_labels.item()['waypoints'][ri]
    pred_confidence = pseudo_labels.item()['confidence'][ri]

    try:
        pred_wp_vo = np.array(pseudo_labels_vo.item()[scene]['waypoints'])
        pred_wp_vo = (M @ pred_wp_vo.T).T
    except:
        scene = None
        continue

    seq_len = 1
    pred_len = 4
    seq_x, seq_y, seq_theta = [], [], []

    for i in range(index, index+seq_len+pred_len):    
        c_img = str(index).zfill(4)
        f_img = str(i).zfill(4)
        with open(scene.replace(c_img, f_img).replace('png','json').replace('rgb_front','measurements'), 'r') as f:
            data = json.load(f)
            seq_x.append(data['x'])
            seq_y.append(data['y'])
            seq_theta.append(data['theta'])
    gt_wp = calc_wp(seq_x, seq_y, seq_theta)

    pred_wp_x = [p[0] for p in pred_wp]
    pred_wp_y = [p[1] for p in pred_wp]

    pred_wp_vo_x = [p[0] for p in pred_wp_vo]
    pred_wp_vo_y = [p[1] for p in pred_wp_vo]

    gt_wp_x = [p[0] for p in gt_wp]
    gt_wp_y = [p[1] for p in gt_wp]

    fig = plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.imshow(Image.open(scene))

    plt.subplot(1,2,2)
    plt.scatter(pred_wp_x, pred_wp_y, label=f"pseudo {pred_confidence}", alpha=0.5)
    plt.scatter(pred_wp_vo_x, pred_wp_vo_y, label=f"pseudo vo", alpha=0.5)
    plt.scatter(gt_wp_x, gt_wp_y, label="gt", alpha=0.5)
    plt.ylim([-15, 0])
    plt.xlim([-7,7])
    plt.legend()

    # plt.show()
    plt.savefig(f'/mnt/qb/work/geiger/pghosh58/transfuser/vis/wp/{save_index}.png')
    save_index += 1