import numpy as np

i=10000
a = np.load('psuedo_waypoints0.npy', allow_pickle=True).item()
while i < 150000:
    print(i)
    b = np.load(f'psuedo_waypoints{i}.npy', allow_pickle=True).item()
    a.update(b)
    i += 10000

np.save(f'psuedo_waypoints.npy', a)
