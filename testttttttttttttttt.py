from torch_geometric.utils import train_test_split_edges

from data_utils import MyOwnDataset
import torch
from torch_geometric.loader import DataLoader
'''
dataset = torch.load('./data/gnn/processed/data.pt')

data = dataset[0]

print("==========train_test_split_edges(data)==========")
data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)

torch.save(data, './data/split_data.pt')

_data = torch.load('./data/split_data.pt')

print("")
print(_data)
'''
import torch

a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([2, 2, 3, 3])
c = torch.tensor([1,1,0,0])