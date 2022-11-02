import datetime
import os
import time

import torch
from torch_geometric.transforms import RandomNodeSplit
from models import GCN, clusterLayer
from metrics import nmi_score,  ari_score, cluster_acc
import losses
import cluster
# 分配GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
devices = [torch.device("cuda:" + str(i)) for i in range(4)]
# for i in range(len(devices)):
    # devices[i] = devices[0]
loss_cuda = devices[0]

print(15*'=' + 'Load Dataset' + 15*'=')
dataset_path = './dataset/gnn/geolife_e2dtc_gru'
dataset_path = './dataset/gnn/geolife_ts_bert'
# data_path = './data/gnn/processed/data.pt'
epochs = 100
hidden_size = 256
n_clusters = 12
alpha = 1

dataset = torch.load(dataset_path + '/processed/data.pt')
data = dataset[0]
print('data_path: {}'.format(dataset_path + '/processed/data.pt'))
print(data)
print()

print(15*'='+'Split Nodes'+15*'=')
transform = RandomNodeSplit(split='train_rest',num_val=0.1,num_test=0.1)
data = transform(data)
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask
edge_index = data.edge_index.to(devices[0])
num_features = data.num_features
x = data.x.to(devices[0])
y = data.y.long().to(devices[0])

train_num = sum(train_mask)
val_num = sum(val_mask)
test_num = sum(test_mask)
print('train_num: {}, val_num: {}, test_num: {}'.format(train_num, val_num, test_num))
print()

# model
model = GCN(hidden_channels=hidden_size)
print(15*'='+'Print Model'+15*'=')
print(model)
print()

# move to GPU (if available)
model = model.to(devices[0])
clusterlayer = clusterLayer(n_clusters, hidden_size, alpha).to(devices[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
# criterion = torch.nn.CrossEntropyLoss().to(devices[0])

def train():
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index)
    # q = output
    q = clusterlayer(output.to(devices[2]))
    p = cluster.target_distribution(q)
    # p_select = p[0]

    pred = q.argmax(1)
    pred_result = pred[train_mask]
    label = y[train_mask]  # Check against ground-truth labels.

    # loss = criterion(out[train_mask], y[train_mask])
    # loss = losses.kl_loss(out, devices[0])
    loss = losses.clusteringLoss(clusterlayer, output, p, devices[2], loss_cuda)
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
        q = clusterlayer(output.to(devices[2]))
        p = cluster.target_distribution(q)

        pred = q.argmax(1)
        pred_result = pred[train_mask]
        label = y[train_mask]  # Check against ground-truth labels.

        loss = losses.clusteringLoss(
            clusterlayer, output, p, devices[2], loss_cuda)
        acc = cluster_acc(label.cpu().numpy(), pred_result.cpu().numpy())  # UACC
        nmi = nmi_score(label.cpu(), pred_result.cpu())
        ari = ari_score(label.cpu(), pred_result.cpu())
    return loss, acc, nmi, ari

def test():
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index)
        q = clusterlayer(output.to(devices[2]))

        pred = q.argmax(dim=1)  # Use the class with highest probability.
        pred_result = pred[train_mask]
        label = y[train_mask]  # Check against ground-truth labels.

        acc = cluster_acc(label.cpu().numpy(), pred_result.cpu().numpy())  # UACC
        nmi = nmi_score(y[test_mask].cpu(), pred[test_mask].cpu())
        ari = ari_score(y[test_mask].cpu(), pred[test_mask].cpu())
    return acc, nmi, ari

print(15*'=' + 'Initiate Clusters' + 15*'=')
print("Time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
start_time = time.time()
cluster.init_cluster(x, clusterlayer, n_clusters, devices[2], dataset_path)
end_time = time.time()
print(f"Time: {end_time-start_time:.2f}s")

print(15*'='+'Start Training'+15*'=')
print("Total Epochs:", epochs)
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
