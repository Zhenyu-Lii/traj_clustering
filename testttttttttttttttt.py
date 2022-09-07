from data_utils import MyOwnDataset
import torch
from torch_geometric.loader import DataLoader

dataset = torch.load('./data/gnn/processed/data.pt')

data = dataset[0]

trainloader = DataLoader(dataset[0],batch_size=64)

print("load finished")