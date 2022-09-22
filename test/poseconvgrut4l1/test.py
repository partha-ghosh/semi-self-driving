import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import outer


def get_rel_pos(abs0, abs1, abs2):
    print("for rel",abs0,abs1,abs2)
    tm1_2_t0 = np.array([abs1[0]-abs0[0], abs1[1]-abs0[1]])
    unit_tm1_2_t0 = tm1_2_t0/np.linalg.norm(tm1_2_t0)
    t0_2_tp1 = np.array([abs2[0]-abs1[0], abs2[1]-abs1[1]])
    x_proj = unit_tm1_2_t0.dot(t0_2_tp1)
    x = unit_tm1_2_t0 * x_proj
    y = t0_2_tp1-x
    y_proj = np.linalg.norm(y)
    if np.cross(x, y) < 0:
        y_proj = -y_proj
    if np.isnan(x_proj):
        x_proj = 0.0
    if np.isnan(y_proj):
        y_proj = 0.0
    return np.array([x_proj, y_proj])

def get_abs_pos(abs0, abs1, rel2=None):
    print('for abs',abs0, abs1, rel2)
    rel1 = abs1 - abs0
    v = np.array([rel1[0], rel1[1]])
    v = v/np.linalg.norm(v)

    r = np.linalg.norm(rel2)
    u = rel2/r
    x_proj = r*(u[0]*v[0]-u[1]*v[1])
    y_proj = r*(u[0]*v[1]+u[1]*v[0]) 

    return abs1+np.array([x_proj,y_proj])

abss = np.array([
 [ 0.00000000e+00,  0.00000000e+00],
 [-1.59175561e-02,  8.01086427e-05],
 [-4.93371748e-01,  2.73895264e-03],
 [-1.76413632e+00,  9.81903078e-03],
 [-3.63857205e+00,  1.88636780e-02],
 [-5.11457677e+00,  2.66075135e-02],
 [-5.20912839e+00,  2.68554688e-02],
 [-5.20931532e+00,  2.68554688e-02],
 [-5.20933439e+00,  2.68554688e-02],
 [-5.20933630e+00,  2.68554688e-02],
 [-5.20933630e+00,  2.68554688e-02],
 [-5.20933630e+00,  2.68554688e-02],
 [-5.21302910e+00,  2.68669129e-02],
 [-5.40789279e+00,  2.85911560e-02],
 [-6.42661922e+00,  3.44886780e-02],
 [-8.11058123e+00,  3.06091309e-02],
 [-1.01257676e+01, -9.08088685e-02],
 [-1.21591987e+01, -5.28816224e-01],
 [-1.39266244e+01, -1.38404846e+00],
 [-1.54038136e+01, -2.75309372e+00],
 [-1.63797105e+01, -4.53166962e+00],
 [-1.68948723e+01, -6.19683457e+00],
 [-1.70668429e+01, -7.94458390e+00],
 [-1.70899787e+01, -1.02229271e+01],
 [-1.70532696e+01, -1.29657631e+01],
 [-1.70900993e+01, -1.58408852e+01],
 [-1.72194774e+01, -1.92027397e+01],
 [-1.73643095e+01, -2.26805764e+01],
 [-1.74285343e+01, -2.61429177e+01],
 [-1.74262926e+01, -2.95585900e+01],
 [-1.73895606e+01, -3.27862664e+01],
 [-1.73770793e+01, -3.38410912e+01],
 [-1.73770674e+01, -3.38424035e+01],
 [-1.73770660e+01, -3.38425255e+01],
 [-1.73774036e+01, -3.39525490e+01],
 [-1.73810549e+01, -3.47378426e+01],
 [-1.73904176e+01, -3.62425652e+01],
 [-1.74028121e+01, -3.82674141e+01],
 [-1.74089273e+01, -4.01796303e+01],
 [-1.74125858e+01, -4.15120659e+01],
 [-1.74127942e+01, -4.15416756e+01],
 [-1.74127942e+01, -4.15418587e+01],
 [-1.74127942e+01, -4.15418816e+01],
 [-1.74127942e+01, -4.15418816e+01],
 [-1.74127942e+01, -4.15418816e+01],
 [-1.74127942e+01, -4.15418816e+01],
 [-1.74127942e+01, -4.15418816e+01],
 [-1.74127942e+01, -4.15418816e+01],
 [-1.74127942e+01, -4.15418816e+01],
 [-1.74127942e+01, -4.15418816e+01],
 [-1.74127942e+01, -4.15418816e+01],
 [-1.74127942e+01, -4.15418816e+01]])

diff = abss[1:] - abss[:-1]
i = 0
while np.linalg.norm(diff[i]) == 0:
    abss = np.delete(abss, 1, 0)
    i+=1

diff = diff[i:]

for i in range(1,len(diff)):
    if np.linalg.norm(diff[i]) == 0:
        diff[i] = diff[i-1]

diff = diff/np.linalg.norm(diff).reshape((-1,1))

i=0
for j in range(len(diff)):
    if (diff[j]==diff[i]).all():
        diff[j] += diff[j-1]
    else:
        i = j

print(diff)

abss[1:] += 0.00001*diff

abs_wp = [
    np.array([ 0.00000000e+00,  0.00000000e+00]),
    np.array([-1.59175561e-02,  8.01086427e-05])]
for i in range(2,len(abss)):
    rel_poss = get_rel_pos(abss[i-2],abss[i-1],abss[i])
    # abs_poss = get_abs_pos(abss[i-2], abss[i-1], rel_poss)
    abs_poss = get_abs_pos(abs_wp[i-2], abs_wp[i-1], rel_poss)

    print('------')
    print(rel_poss)
    print(abs_poss)
    print(abss[i])
    print('------')

    abs_wp.append(abs_poss)

abs_wp = np.array(abs_wp)

plt.figure()
plt.plot(abs_wp[:,0], abs_wp[:,1], alpha=0.5, linewidth=2, marker='.', label='abs_wp')
plt.plot(abss[:,0], abss[:,1], alpha=0.5, marker='.', label='abs_pos')
plt.legend()

# plt.scatter(rel_x, rel_y, alpha=0.3)
plt.savefig('y.png')