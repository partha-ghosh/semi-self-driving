import os
from os import listdir
from os.path import isfile, join, isdir
import cv2
import os
import tqdm


def getAllFilesRecursive(root):
    files = [ join(root,f) for f in listdir(root) if isfile(join(root,f))]
    dirs = [ d for d in listdir(root) if isdir(join(root,d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root,d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root,f))
    return files

files = getAllFilesRecursive('/mnt/qb/work/geiger/pghosh58/transfuser/data/14_weathers_minimal_data')
# files = getAllFilesRecursive('/mnt/qb/work/geiger/pghosh58/transfuser/data/transfuser_plus_data')
# files = getAllFilesRecursive('/mnt/qb/geiger/kchitta31/datasets/carla/pami_v1_dataset_23_11')

images = []
for f in files:
    if '.png' in f:
        if 'rgb_front' in f:
            images.append(f)


video_name = 'video1.mp4'

frame = cv2.imread(images[0])
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter(video_name, fourcc, 18, (width,height))

for image in tqdm.tqdm(images):
    video.write(cv2.resize(cv2.imread(image), (width,height)))

cv2.destroyAllWindows()
video.release()