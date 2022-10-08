from utils import *


# data_path = '/mnt/qb/work/geiger/pghosh58/transfuser/data/14_weathers_minimal_data'
data_path = '/mnt/qb/work/geiger/pghosh58/transfuser/data/new'
# data_path = '/mnt/qb/work/geiger/pghosh58/transfuser/data/transfuser_plus_data'
# data_path = '/mnt/qb/geiger/kchitta31/datasets/carla/pami_v1_dataset_23_11'
dest_path = f'/mnt/qb/work/geiger/pghosh58/transfuser/data/filtered_{data_path.split("/")[-1]}'
print(dest_path)
seq_len = 1
pred_len = 4


towns = next(os.walk(data_path))[1]

jobs = []
for town in tqdm.tqdm(towns, desc='town'):
    routes = next(os.walk(f'{data_path}/{town}'))[1]
    # worker(town, routes, data_path, seq_len, pred_len, dest_path)
    p = mp.Process(target=worker, args=(town, routes, data_path, seq_len, pred_len, dest_path))
    jobs.append(p)
    p.start()
for p in jobs:
    p.join()


for town in tqdm.tqdm(towns, desc='town'):
    data = dict(
        turns=[],
        in_motion=[],
        long_stops=[],
        short_stops=[]
    )
    town_path = f'{dest_path}/{town}/processed_data.npy'
    filter_data(town_path, data)
    np.save(f'{dest_path}/{town}/filtered_data.npy', data)
