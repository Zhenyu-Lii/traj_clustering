import datetime
import os
import time
import torch
import losses
import cluster
import argparse
from torch_geometric.transforms import RandomNodeSplit
from models import GCN, clusterLayer
from metrics import nmi_score,  ari_score, cluster_acc

parser = argparse.ArgumentParser()

parser.add_argument("-n_clusters", type=int, default=20,
                    help="Number of luster")

parser.add_argument("-hidden_size", type=int, default=256,
                    help="The hidden state size in GCNConv")


parser.add_argument("-epoch", type=int, default=100,
                    help="The training epoch")

parser.add_argument("-alpha", type=int, default=1)

parser.add_argument("-update_interval", type=int, default=1,
                    help="update interval of model")

parser.add_argument("-dataset_path", type=str, default='./dataset/gnn/geolife_e2dtcF_bert_1107',
                    help="update interval of model")

args = parser.parse_args()
# 分配GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
devices = [torch.device("cuda:" + str(i)) for i in range(6)]
# for i in range(len(devices)):
    # devices[i] = devices[1]
loss_cuda = devices[1]

print(15*'=' + 'Load Dataset' + 15*'=')
dataset_path = args.dataset_path
# dataset_path = './dataset/gnn/geolife_ts_bert'
# data_path = './data/gnn/processed/data.pt'
epochs = args.epoch
alpha = args.alpha
update_interval = args.update_interval
hidden_size = args.hidden_size
n_clusters = args.n_clusters


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
edge_index = data.edge_index.to(devices[1])
num_features = data.num_features
x = data.x.to(devices[1])
y = data.y.long().to(devices[1])

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
model = model.to(devices[1])
clusterlayer = clusterLayer(args, alpha).to(devices[0])
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
# criterion = torch.nn.CrossEntropyLoss().to(devices[1])

def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index)
    # output = x
    if epoch % update_interval == 0:
        with torch.no_grad():
            tmp_q = clusterlayer(output.to(devices[0]))
            p = cluster.target_distribution(tmp_q)
        # p_select = p[0]

            pred = tmp_q.argmax(1)
            pred_result = pred[train_mask]
            label = y[train_mask]  # Check against ground-truth labels.

            train_acc = cluster_acc(label.cpu().numpy(), pred_result.cpu().numpy())  # UACC
            nmi = nmi_score(label.cpu(), pred_result.cpu())
            ari = ari_score(label.cpu(), pred_result.cpu())
            print(f'Train Acc: {train_acc:.4f}, Train NMI: {nmi:.4f}, Train ARI: {ari:.4f}')

            # loss = criterion(out[train_mask], y[train_mask])
                # loss = losses.kl_loss(out, devices[1])
    loss = losses.clusteringLoss(clusterlayer, output, p, devices[0], loss_cuda)
    loss = loss.requires_grad_()
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    print(f'Train Loss: {loss:.4f}')
    return loss

def val():
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index)
        q = clusterlayer(output.to(devices[0]))
        p = cluster.target_distribution(q)

        pred = q.argmax(1)
        pred_result = pred[train_mask]
        label = y[train_mask]  # Check against ground-truth labels.

        loss = losses.clusteringLoss(
            clusterlayer, output, p, devices[0], loss_cuda)
        acc = cluster_acc(label.cpu().numpy(), pred_result.cpu().numpy())  # UACC
        nmi = nmi_score(label.cpu(), pred_result.cpu())
        ari = ari_score(label.cpu(), pred_result.cpu())
    return loss, acc, nmi, ari

def test():
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index)
        q = clusterlayer(output.to(devices[0]))

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
cluster.init_cluster(x, clusterlayer, n_clusters, devices[0], dataset_path)

tmp_q = clusterlayer(x.to(devices[0]))
p = cluster.target_distribution(tmp_q)
# p_select = p[0]

pred = tmp_q.argmax(1)
pred_result = pred[train_mask]
label = y[train_mask]  # Check against ground-truth labels.

train_acc = cluster_acc(label.cpu().numpy(), pred_result.cpu().numpy())  # UACC
nmi = nmi_score(label.cpu(), pred_result.cpu())
ari = ari_score(label.cpu(), pred_result.cpu())
print(f'init Acc: {train_acc:.4f}, init NMI: {nmi:.4f}, init ARI: {ari:.4f}')

end_time = time.time()
print(f"Time: {end_time-start_time:.2f}s")

print(15*'='+'Start Training'+15*'=')
print("Total Epochs:", epochs)
for epoch in range(0, epochs):
    print(f'Epoch: {epoch + 1:03d}')
    print("Time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    train_loss = train(epoch)
    val_loss, val_acc, val_nmi, val_ari = val()
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val NMI: {val_nmi:.4f}, Val ARI: {val_ari:.4f}')

acc, nmi, ari = test()
print(15*'=' + 'Test Result' + 15*'=')
print(f'test_acc: {acc:.4f}, test_nmi: {nmi:.4f}, test_ari: {ari:.4f}')

torch.save(model,'./models/GAE.pt')
