# Install required packages.
import os
import torch
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.utils import train_test_split_edges

from models import GCNEncoder
from models import GCN
from torch_geometric.nn import GAE
from torch_geometric.nn import GCNConv
from metrics import nmi_score,  ari_score
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
# data_path = './data/split_data.pt'
# data_path = './feature_learning/bert/data/gnn/geolife_ts/processed/split_edges_data.pt'
data_path = './feature_learning/bert/data/gnn/geolife_ts/processed/data.pt'

print(15*'='+'Load Dataset'+15*'=')
data = torch.load(data_path)
print(data)
print()
'''
train_mask = torch.tensor([True if i in data.train_pos_edge_index[0] else False for i in range(0,data.num_nodes)])
val_mask = torch.tensor([True if i in data.val_pos_edge_index[0] else False for i in range(0,data.num_nodes)])
test_mask =torch.tensor([True if i in data.test_pos_edge_index[0] else False for i in range(0,data.num_nodes)])
'''
print(15*'='+'Split Nodes'+15*'=')
transform = RandomNodeSplit(split='train_rest',num_val=0.1,num_test=0.1)
data = transform(data)
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask
print(data)
print()

train_num = sum(train_mask)
val_num = sum(val_mask)
test_num = sum(test_mask)
print('train_num: {}, val_num: {}, test_num: {}'.format(train_num, val_num, test_num))

x = data.x.to(device)
y = data.y.long().to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
val_pos_edge_index = data.val_pos_edge_index.to(device)
test_pos_edge_index = data.test_pos_edge_index.to(device)
# parameters
out_channels = 2
num_features = data.num_features
epochs = 200

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
print(15*'='+'Print Model'+15*'=')
print(model)

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
    recon_loss = model.gaeLayer.recon_loss(z, train_pos_edge_index)
    classify_loss = criterion(out[train_mask], y[train_mask])
    #if args.variational:
    loss = recon_loss + classify_loss
    #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1)  # Use the class with highest probability.
    pred_result = pred[train_mask]
    label = y[train_mask]  # Check against ground-truth labels.
    train_correct = pred_result.eq(label)
    tmp1 = train_correct.sum()
    tmp2 = train_mask.long().sum()
    train_acc = int(tmp1) / int(tmp2)  # Derive ratio of correct predictions.\
    nmi = nmi_score(label.cpu(), pred_result.cpu())
    ari = ari_score(label.cpu(), pred_result.cpu())

    return recon_loss, classify_loss, train_acc, nmi, ari


def test():
    model.eval()
    with torch.no_grad():
        z = model.gaeLayer.encode(x, test_pos_edge_index)
        out = model.classifyLayer(z, test_pos_edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[test_mask].eq(y[test_mask])  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(sum(test_mask))  # Derive ratio of correct predictions.
        nmi = nmi_score(y[test_mask].cpu(), pred[test_mask].cpu())
        ari = ari_score(y[test_mask].cpu(), pred[test_mask].cpu())
    return acc, nmi, ari


def val():
    model.eval()
    with torch.no_grad():
        z = model.gaeLayer.encode(x, val_pos_edge_index)
        out = model.classifyLayer(z, val_pos_edge_index)
        recon_loss = model.gaeLayer.recon_loss(z, val_pos_edge_index)
        classify_loss = criterion(out[val_mask], y[val_mask])
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[val_mask].eq(y[val_mask])  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(sum(val_mask))  # Derive ratio of correct predictions.
        nmi = nmi_score(y[val_mask].cpu(), pred[val_mask].cpu())
        ari = ari_score(y[val_mask].cpu(), pred[val_mask].cpu())
    return recon_loss, classify_loss, acc, nmi, ari

from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset

print(15*'='+'Start Training'+15*'=')
print("Total Epochs:", epochs)
for epoch in range(1, epochs+1):

    train_recon_loss, train_classify_loss, train_acc, train_nmi, train_ari = train()
    val_recon_loss, val_classify_loss, val_acc, val_nmi, val_ari= val()
    print(f'Epoch: {epoch:03d}\nTrain Loss: {train_recon_loss+train_classify_loss:.4f}, Train Acc: {train_acc:.4f}, Train NMI: {train_nmi:.4f}, Train ARI: {train_ari:.4f}, recon_loss: {train_recon_loss:.4f}, cross_entropy_loss: {train_classify_loss:.4f}'
          f'\nVal Loss: {val_recon_loss+val_classify_loss:.4f}, Val Acc: {val_acc:.4f}, Val NMI: {val_nmi:.4f}, Val ARI: {val_ari:.4f}, recon_loss: {val_recon_loss:.4f}, cross_entropy_loss: {val_classify_loss:.4f}')

acc, nmi, ari = test()
print(15*'=' + 'Test Result' + 15*'=')
print(f'test_acc: {acc:.4f}, test_nmi: {nmi:.4f}, test_ari: {ari:.4f}')

torch.save(model,'./models/GAE.pt')
