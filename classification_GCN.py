# Install required packages.
import argparse
import datetime
import os
import torch
from visualize.functions import visualize
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.transforms import RandomNodeSplit
from models import GCN, clusterLayer
from metrics import nmi_score,  ari_score, cluster_acc
import losses
import cluster

os.environ['TORCH'] = torch.__version__
print(f'torch version: {torch.__version__}')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 分配GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
devices = [torch.device("cuda:" + str(i)) for i in range(4)]
# for i in range(len(devices)):
    # devices[i] = devices[0]
loss_cuda = devices[0]

'''
def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()
'''

data_path = './dataset/gnn/geolife_ts_bert/processed/data.pt'
epochs = 100
hidden_size = 256
n_clusters = 12
alpha = 1


# data_path = './data/gnn/processed/data.pt'
dataset = torch.load(data_path)
data = dataset[0]
print(f'data_path: {data_path}')
print(data)
print()

print(15*'='+'Split Nodes'+15*'=')
transform = RandomNodeSplit(split='train_rest',num_val=0.1,num_test=0.1)
data = transform(data)
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask

train_num = sum(train_mask)
val_num = sum(val_mask)
test_num = sum(test_mask)
print('train_num: {}, val_num: {}, test_num: {}'.format(train_num, val_num, test_num))

x = data.x.to(devices[0])
y = data.y.long().to(devices[0])
edge_index = data.edge_index.to(devices[0])

num_features = data.num_features

# model
model = GCN(hidden_channels=hidden_size)
print(15*'='+'Print Model'+15*'=')
print(model)

# move to GPU (if available)
model = model.to(devices[0])
clusterlayer = clusterLayer(n_clusters, hidden_size, alpha).to(devices[2])
# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# criterion = torch.nn.CrossEntropyLoss().to(devices[0])
def train():
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index)

    q = output
    q = clusterlayer(x.to(devices[2]))
    p = cluster.target_distribution(q)
    # p_select = p[0]

    pred = output.argmax(1)
    pred_result = pred[train_mask]
    label = y[train_mask]  # Check against ground-truth labels.

    # loss = criterion(out[train_mask], y[train_mask])
    # loss = losses.kl_loss(out, devices[0])
    loss = losses.clusteringLoss(clusterlayer, output, p, q, devices[2], loss_cuda)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    # evaluate clustering performance
    # train_correct = pred[train_mask] == y[train_mask]  # Check against ground-truth labels.
    # train_acc = (int(train_correct.sum()) / int(train_mask.sum())  # Derive ratio of correct predictions.
    train_acc = cluster_acc(label.cpu().numpy(), pred_result.cpu().numpy())  # UACC
    nmi = nmi_score(label.cpu(), pred_result.cpu())
    ari = ari_score(label.cpu(), pred_result.cpu())

    return loss, train_acc, nmi, ari

def val():
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index)
        q = output
        p = cluster.target_distribution(q)

        pred = output.argmax(1)
        pred_result = pred[train_mask]
        label = y[train_mask]  # Check against ground-truth labels.
        loss = losses.clusteringLoss(
            clusterlayer, output, p, q, devices[2], loss_cuda)

        correct = pred[val_mask].eq(y[val_mask])  # Check against ground-truth labels.
        acc = cluster_acc(label.cpu().numpy(), pred_result.cpu().numpy())  # UACC
        nmi = nmi_score(label.cpu(), pred_result.cpu())
        ari = ari_score(label.cpu(), pred_result.cpu())
    return loss, acc, nmi, ari

def test():
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[test_mask].eq(y[test_mask])  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(sum(test_mask))  # Derive ratio of correct predictions.
        nmi = nmi_score(y[test_mask].cpu(), pred[test_mask].cpu())
        ari = ari_score(y[test_mask].cpu(), pred[test_mask].cpu())
    return acc, nmi, ari

print(15*'='+'Start Training'+15*'=')
print("Total Epochs:", epochs)

cluster.init_cluster(x, clusterlayer, n_clusters, devices[2])
for epoch in range(1, epochs+1):
    train_loss, train_acc, train_nmi, train_ari = train()
    val_loss, val_acc, val_nmi, val_ari = val()
    print(f'Epoch: {epoch:03d}')
    print("Time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train NMI: {train_nmi:.4f}, Train ARI: {train_ari:.4f}'
          f'\nVal Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val NMI: {val_nmi:.4f}, Val ARI: {val_ari:.4f}')

acc, nmi, ari = test()
print(15*'=' + 'Test Result' + 15*'=')
print(f'test_acc: {acc:.4f}, test_nmi: {nmi:.4f}, test_ari: {ari:.4f}')

torch.save(model,'./models/GAE.pt')
