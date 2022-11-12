import pandas as pd
import numpy as np
from collections import Counter
import torch

save = True
dataset = 'e2dtcF'
file_name = 'data'
data_path = f'./traj/{dataset}/{file_name}.h5'
if (dataset == 'cdr'):
    loc_index_cdr = pd.read_csv('../preprocess/CDR/processed/loc_index_cdr_582.csv')
    replace_dict = dict(zip(loc_index_cdr['loc_index'], loc_index_cdr['new_index']))

# 统计一共有多少个token
def get_max_id(data_path):
    trajs = pd.read_hdf(data_path)['trajectory']
    tokens = []
    for i, traj in enumerate(trajs):
        print(f'Traj {i}')
        point_list = list(filter(lambda x: x != '[PAD]', traj.split(' ')))
        if dataset == 'cdr':
            point_list = list(map(lambda x: replace_dict[int(x)]+4, point_list))
        else:
            point_list = list(map(lambda x: int(x), point_list))
        tokens = tokens + point_list
    tokens = np.unique(tokens)
    max_id = np.array(tokens).max() # 全矩阵最大值
    min_id = np.array(tokens).min()
    length = len(np.array(tokens))
    print(f'max_id: {max_id}, min_id: {min_id}, length: {length}')
    return max_id, min_id, length

def gen_word_vector(length, traj):
    word_vector = length*[0]
    traj_list = list(filter(lambda x: x != '[PAD]', traj.split(' ')))
    if dataset == 'cdr':
        point_list = list(map(lambda x: replace_dict[int(x)], traj_list)) # cdr需要replace index
    else:
        point_list = list(map(lambda x: int(x), traj_list))
    # point_list = [int(i) for i in traj.split(' ') if i != '[PAD]']
    cnt = dict(Counter(point_list))
    for key, value in zip(cnt.keys(), cnt.values()):
        if dataset == 'cdr':
            word_vector[key] = value
        elif dataset == 'e2dtcF':
            word_vector[key-4] = value
    # print(f'traj_count: {cnt}')
    # print(f'word_vectory: {word_vector}')
    return word_vector

data = pd.read_hdf(data_path)
labels = np.array(data['label']).T
max_id, min_id, length = get_max_id(data_path)

if save == True:
    print('==>generate word list...')
    word_list = [gen_word_vector(length, traj) for traj in data['trajectory']]
# for traj in data['trajectory']:
#     word_list.append(gen_word_vector(max_id, traj))

    torch.save(torch.tensor(word_list), f'./traj/{dataset}/embeddings/bow.pt')

    np.savetxt(f'./for_sdcn/{dataset}/{dataset}.txt', word_list, fmt='%d')
    np.savetxt(f'./for_sdcn/{dataset}/{dataset}_label.txt', labels, fmt='%d')

    print('==>word list saved.')
    print(f'save_path: ./traj/{dataset}/embeddings/bow.pt')


