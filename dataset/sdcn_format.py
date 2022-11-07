import datetime

import torch
import pandas as pd
import numpy as np
import os
import time
from data_utils import MyOwnDataset
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from torch_geometric.utils import train_test_split_edges

# init embeddings
# loc_index = pd.read_csv('../../data/loc_index.csv',index_col=0)
# loc_size = len(loc_index)
embed_size = 256
sgm = 0.8
lbd = 0.6
cwd = os.path.abspath('.')
# embed_256 = np.random.randn(loc_size, embed_size)
# np.save('data/Geolife/embedding_256.npy', embed_256)

# Geolife New
# traj_path = cwd+"/all_traj_labeled_σ_{sgm}_λ_{lbd}.h5".format(sgm=sgm, lbd=lbd)
# embed_path = './data/Geolife/embed_256.pt'
# all_traj = pd.read_hdf(traj_path, key='data')
# all_data = all_traj.drop(all_traj[all_traj['label']==-1].index) # drop掉label为-1的轨迹
# labels = np.array(all_data['label'])

# E2DTC
dataset = 'e2dtcF'
embed_name = 'e2dtcF_bow'

traj_path = f'./traj/{dataset}/data.h5'
data_path = f'./traj/{dataset}/{embed_name}.pt'

word_list = torch.load(data_path).numpy()
all_traj = pd.read_hdf(traj_path)
labels = np.array(all_traj['label']).T

np.savetxt(f'./for_sdcn/{dataset}/{embed_name}.txt', word_list, fmt='%d')
np.savetxt(f'./for_sdcn/{dataset}/{embed_name}_label.txt', labels, fmt='%d')