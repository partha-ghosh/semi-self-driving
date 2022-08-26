import os

os.system('conda install -y -c anaconda networkx ephem requests numpy')
os.system('conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch')
os.system('pip install py-trees==0.8.3 dictor tensorboard opencv-python-headless')
os.system('conda install -y -c conda-forge tabulate shapely tqdm libjpeg-turbo imgaug')