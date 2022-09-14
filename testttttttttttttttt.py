from torch_geometric.transforms import RandomNodeSplit
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
'''
print("==========train_test_split_edges(data)==========")
data = torch.load('./data/split_data.pt')
print("load finished.\n")
train_mask = torch.tensor([True if i in data.train_pos_edge_index[0] else False for i in range(0,data.num_nodes)])
val_mask = torch.tensor([True if i in data.val_pos_edge_index[0] else False for i in range(0,data.num_nodes)])
test_mask =torch.tensor([True if i in data.test_pos_edge_index[0] else False for i in range(0,data.num_nodes)])

torch.save(train_mask,'./data/gnn/processed/train_mask.pt')
torch.save(val_mask,'./data/gnn/processed/val_mask.pt')
torch.save(test_mask,'./data/gnn/processed/test_mask.pt')
'''
'''
train_mask = torch.load('./data/gnn/processed/train_mask.pt')
val_mask = torch.load('./data/gnn/processed/val_mask.pt')
test_mask = torch.load('./data/gnn/processed/test_mask.pt')

print(train_mask)
print(train_mask.sum())
print(val_mask)
print(val_mask.sum())
print(test_mask)
print(test_mask.sum())
tmp1 = train_mask & val_mask
tmp2 = train_mask & test_mask
print(tmp1.sum())
print(tmp2.sum())
'''
data = torch.load('./data/split_data.pt')

transform = RandomNodeSplit(split='train_rest',num_val=0.1,num_test=0.1)
data = transform(data)
print(data)