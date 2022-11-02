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
traj_path = './data/E2DTC/data.h5'
embed_path = './data/E2DTC/embed_256_bert50.pt'
embed_path = './data/E2DTC/maxlen_79'
# dataset_path = './data/gnn/geolife_e2dtc'
dataset_path = './data/gnn/geolife_ts_79'

all_traj = pd.read_hdf(traj_path)
labels = np.array(all_traj['label'])

print(f'current working directory: {cwd}\ndata_file: {traj_path}\nembed_file: {embed_path}\nsaving_path: {dataset_path}')
print("Time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(5*'='+'>', 'Generate KNN Graph...')

start_time = time.time()
embeddings = torch.load(embed_path + '/embed_256_batch_0.pt')
# concat embeddings
for i in range (1,259):
    embedding = torch.load(embed_path + f'/embed_256_batch_{i}.pt')
    embeddings = torch.cat((embeddings, embedding), dim=0)

embeddings = embeddings.cpu().detach().numpy()
print('embedding shape:', embeddings.shape)
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(embeddings)
distances, indices = nbrs.kneighbors(embeddings)
A = nbrs.kneighbors_graph(embeddings).toarray()

edge_index, _ = from_scipy_sparse_matrix(coo_matrix(A))
end_time = time.time()
print(f"KNN Graph: {end_time-start_time:.2f}s")

print("-" * 7 + "edge index" + "-" * 7)
print(edge_index)
'''
构建GNN的数据集
'''
# 放入datalist
print(5*'='+'>', "Generate Dataset...")
dataset = MyOwnDataset(dataset_path, edge_index, embeddings, labels)
data = dataset[0]

print("Time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(5*'='+'>', 'train_test_split_edges(data)...')
start_time = time.time()
data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)
print(data)
end_time = time.time()
print(f"Split edges: {end_time-start_time:.2f}s")

torch.save(data, dataset_path + '/processed/split_edges_data.pt')