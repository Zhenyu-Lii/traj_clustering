import torch
import pandas as pd
import numpy as np
import os
from data_utils import DataOrderScaner, load_label
from data_utils import MyOwnDataset
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from torch_geometric.utils import train_test_split_edges

# init embeddings
loc_index = pd.read_csv('../../data/loc_index.csv',index_col=0)
loc_size = len(loc_index)
embed_size = 256
sgm = 0.8
lbd = 0.6
cwd = os.path.abspath('.')
embed_256 = np.random.randn(loc_size, embed_size)
np.save('data/Geolife/embedding_256.npy', embed_256)

# drop掉label为-1的轨迹
all_traj = pd.read_hdf(cwd+"/all_traj_labeled_σ_{sgm}_λ_{lbd}.h5".format(sgm=sgm, lbd=lbd), key='data')
all_data = all_traj.drop(all_traj[all_traj['label']==-1].index)

labels = np.array(all_data['label'])
embeddings = torch.load('./data/Geolife/embed_256.pt').cpu().detach().numpy()

print(15*'=', 'Generate KNN Graph', 15*'=')
nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(embeddings)
distances, indices = nbrs.kneighbors(embeddings)
A = nbrs.kneighbors_graph(embeddings).toarray()

edge_index, _ = from_scipy_sparse_matrix(coo_matrix(A))
print("-" * 7 + "Dataset done!" + "-" * 7)
print("-" * 7 + "edge index!" + "-" * 7)
print(edge_index)
'''
构建GNN的数据集
'''
# 放入datalist
dataset = MyOwnDataset('./data/gnn/geolife_ts', edge_index, embeddings, labels)
data = dataset[0]

print("==========train_test_split_edges(data)==========")
data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)
print(data)

torch.save(data, './data/gnn/geolife_ts/processed/split_edges_data.pt')
