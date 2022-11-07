import pandas as pd
import numpy as np
from collections import Counter
import torch

dataset = 'e2dtcF'
data_path = f'./traj/{dataset}/data.h5'

# 统计一共有多少个token
def get_max_id(data_path):
    data = pd.read_hdf(data_path)
    trajs = []
    for i in range(len(data)):
        if i % 10000 == 0:
            print(f'Traj {i}')
        traj = data['trajectory'][i].split(' ')
        trajs.append([int(i) if i != '[PAD]' else -1 for i in traj])
    max_id = np.array(trajs).max()
    min_id = np.array(trajs).min()
    print(f'max_id: {max_id},min_id: {min_id}')
    return max_id, min_id

def gen_word_vector(max_id, traj):
    word_vector = max_id*[0]
    point_list = [int(i) for i in traj.split(' ') if i != '[PAD]']
    cnt = dict(Counter(point_list))
    for key, value in zip(cnt.keys(), cnt.values()):
        word_vector[key-1] = value
    # print(f'traj_count: {cnt}')
    # print(f'word_vectory: {word_vector}')
    return word_vector

data = pd.read_hdf(data_path)
max_id, min_id = get_max_id(data_path)

print('==>generate word list...')
word_list = [gen_word_vector(max_id, traj) for traj in data['trajectory']]
# for traj in data['trajectory']:
#     word_list.append(gen_word_vector(max_id, traj))

torch.save(torch.tensor(word_list), f'./traj/{dataset}/bow.pt')
print('==>word list saved.')
print(f'save_path: ./traj/{dataset}/bow.pt')


