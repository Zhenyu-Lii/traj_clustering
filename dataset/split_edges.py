import torch
import datetime
import os
import time
from torch_geometric.utils import train_test_split_edges
data_path = './gnn/geolife_e2dtc_gru/processed/data.pt'
dataset_path = './gnn/geolife_e2dtc_gru'

print(f'data_path: {data_path}')
dataset = torch.load(data_path)
data = dataset[0]

print("Time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(5*'='+'>', 'train_test_split_edges(data)...')
start_time = time.time()
data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)
print(data)
end_time = time.time()
print(f"Split edges time: {end_time-start_time:.2f}s")

torch.save(data, dataset_path + '/processed/split_edges_data.pt')