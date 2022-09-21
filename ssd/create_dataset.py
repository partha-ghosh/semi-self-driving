from utils import *


# data_path = '/mnt/qb/work/geiger/pghosh58/transfuser/data/14_weathers_minimal_data'
# data_path = '/mnt/qb/work/geiger/pghosh58/transfuser/data/transfuser_plus_data'
data_path = '/mnt/qb/geiger/kchitta31/datasets/carla/pami_v1_dataset_23_11'
dest_path = f'/mnt/qb/work/geiger/pghosh58/transfuser/data/filtered_{data_path.split("/")[-1]}'
print(dest_path)
seq_len = 1
pred_len = 4

# determine the innermost dir
i=0
is_not_routes = True
while is_not_routes:
    routes = glob.glob(data_path+("/**"*i))
    is_not_routes = not any(['rgb' in route for route in routes])
    i += 1
# towns = next(os.walk(data_path))[1]
towns = {
    'Town01':[],
    'Town02':[],
    'Town03':[],
    'Town04':[],
    'Town05':[],
    'Town06':[],
    'Town07':[],
    'Town10':[],
}

for route in routes:
    if 'rgb' in route:
        for town in towns:
            if town in route:
                towns[town].append(route)

jobs = []
for town in tqdm.tqdm(towns, desc='town'):
    # routes = next(os.walk(f'{data_path}/{town}'))[1]
    routes = towns[town]
    # worker(town, routes, seq_len, pred_len, dest_path)
#     p = mp.Process(target=worker, args=(town, routes, seq_len, pred_len, dest_path))
#     jobs.append(p)
#     p.start()
# for p in jobs:
#     p.join()


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
