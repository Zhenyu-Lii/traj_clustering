from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch.nn import Linear
from scipy.sparse import coo_matrix
import numpy as np

_A = np.array([[1, 0, 0],[0,1,0],[0,0,1]])
print(_A,'\n')
A = coo_matrix(_A)
print(A,'\n')
tst = bool(A == A.tocoo())
print(A)

class GCN_Net(nn.Module):
    def __init__(self, hidden):
        super(GCN_Net, self).__init__()

        self.conv1 = GCNConv(256, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, 12)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.selu()
        x = self.conv2(x, edge_index)
        x = F.selu()
        x = self.conv3(x, edge_index)


        return F.log_softmax(x, dim=1)
model = GCN_Net(hidden=64)
print(model,'\n')

'''

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]
data'''



class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(1433, hidden_channels)
        self.lin2 = Linear(hidden_channels, 7)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

model = MLP(hidden_channels=16)
print(model)