# Install required packages.
import os
import torch
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import train_test_split_edges

from models import GCNEncoder
from models import GCN
from torch_geometric.nn import GAE
from torch_geometric.nn import GCNConv

os.environ['TORCH'] = torch.__version__
print(torch.__version__)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
print("==========RandomLinkSplit==========")
transform = RandomLinkSplit(is_undirected=True)
train_data, val_data, test_data = transform(data)
train_mask = [1 if i in train_data.edge_label_index[0] else 0 for i in range(0,data.num_nodes)]
val_mask = [1 if i in val_data.edge_label_index[0] else 0 for i in range(0,data.num_nodes)]
train_num = sum(train_mask)
val_num = sum(val_mask)

x = train_data.x.to(device)
y = data.y.long().to(device)
train_pos_edge_index = train_data.edge_label_index.to(device)
val_pos_edge_index = val_data.edge_label_index.to(device)
'''

print("==========train_test_split_edges(data)==========")
data = torch.load('./data/split_data.pt')
print("load finished.\n")
train_mask = torch.tensor([True if i in data.train_pos_edge_index[0] else False for i in range(0,data.num_nodes)])
val_mask = torch.tensor([True if i in data.val_pos_edge_index[0] else False for i in range(0,data.num_nodes)])
test_mask =torch.tensor([True if i in data.test_pos_edge_index[0] else False for i in range(0,data.num_nodes)])

train_num = sum(train_mask)
val_num = sum(val_mask)
test_num = sum(test_mask)
print('train_num: {}, val_num: {}, test_num: {}'.format(train_num, val_num, test_num))

x = data.x.to(device)
y = data.y.long().to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
val_pos_edge_index = data.val_pos_edge_index.to(device)

# parameters
out_channels = 2
num_features = data.num_features
epochs = 100

# model
class MODEL(torch.nn.Module):
    def __init__(self, num_features=256, out_channels=2, hidden_channels=64):
        super().__init__()
        torch.manual_seed(1234567)
        self.gaeLayer = GAE(GCNEncoder(num_features, out_channels))
        self.classifyLayer = GCN(out_channels, hidden_channels)

    def forward(self, x, edge_index):
        z = self.gaeLayer.encoder.forward(x, edge_index)
        c = self.classifyLayer.forward(z, edge_index)

        return c

model = MODEL(num_features, out_channels, hidden_channels=64)

# move to GPU (if available)
model = model.to(device)

# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss().to(device)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.gaeLayer.encode(x, train_pos_edge_index)
    out = model.classifyLayer(z, train_pos_edge_index)
    # Compute the loss solely based on the training nodes.
    loss1 = model.gaeLayer.recon_loss(z, train_pos_edge_index)
    loss2 = criterion(out[train_mask], y[train_mask])
    #if args.variational:
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)  # Use the class with highest probability.
    pred_result = pred[train_mask]
    label = y[train_mask]  # Check against ground-truth labels.
    train_correct = pred_result.eq(label)
    tmp1 = train_correct.sum()
    tmp2 = train_mask.long().sum()
    train_acc = int(tmp1) / int(tmp2)  # Derive ratio of correct predictions.\

    return loss, train_acc


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def val():
    model.eval()
    with torch.no_grad():
        z = model.gaeLayer.encode(x, val_pos_edge_index)
        out = model.classifyLayer(z, val_pos_edge_index)
        loss = model.gaeLayer.recon_loss(z, val_pos_edge_index) + criterion(out[val_mask], y[val_mask])
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[val_mask].eq(y[val_mask])  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(sum(val_mask))  # Derive ratio of correct predictions.
    return loss, acc

from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset


for epoch in range(1, 200):

    train_loss, train_acc = train()
    val_loss, val_acc = val()
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train_acc: {train_acc:.4f}, Val_acc: {val_acc:.4f}')

torch.save(model,'./models/GAE.pt')
