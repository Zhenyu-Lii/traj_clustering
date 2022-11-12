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
dataset = 'cdr'
datafile = 'data_cdr'
pretrain_model = 'gru'
embed_name = f'cdr_newIndex585_gru_epoch_20'

traj_path = f'./traj/{dataset}/{datafile}.h5'
data_path = f'./pretrain_embeddings/{pretrain_model}/{embed_name}.pt'

vecs = torch.load(data_path).numpy()
vecs = (10*vecs).round()
all_traj = pd.read_hdf(traj_path)
labels = np.array(all_traj['label']).T

dirs = f'./for_sdcn/{dataset}/'

if not os.path.exists(dirs):
    os.makedirs(dirs)

np.savetxt(f'./for_sdcn/{dataset}/{embed_name}.txt', vecs, fmt='%d')
np.savetxt(f'./for_sdcn/{dataset}/{embed_name}_label.txt', labels, fmt='%d')

# copy files to Projects/SDCN/data