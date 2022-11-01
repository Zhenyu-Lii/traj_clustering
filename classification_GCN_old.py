# Install required packages.
import argparse
import os
import torch
from visualize.functions import visualize
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.transforms import RandomNodeSplit
from models import GCN, clusterLayer
from metrics import nmi_score,  ari_score
import losses

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
# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss().to(devices[0])

def train():
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    pred = out.argmax(dim=1)  # Use the class with highest probability.
    train_correct = pred[train_mask] == y[train_mask]  # Check against ground-truth labels.
    train_acc = int(train_correct.sum()) / int(train_mask.sum())  # Derive ratio of correct predictions.
    nmi = nmi_score(y[train_mask].cpu(), pred[train_mask].cpu())
    ari = ari_score(y[train_mask].cpu(), pred[train_mask].cpu())

    return loss, train_acc, nmi, ari

def val():
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        # loss = kl_loss(out, devices[0])
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = pred[val_mask].eq(y[val_mask])  # Check against ground-truth labels.
        acc = int(correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
        nmi = nmi_score(y[val_mask].cpu(), pred[val_mask].cpu())
        ari = ari_score(y[val_mask].cpu(), pred[val_mask].cpu())
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
# ========================K-Fold Validation=============================
'''
k=10
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(data.y)))):
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    print('Fold {}'.format(fold + 1))
    train_mask = [1 if i in train_idx else 0 for i in range(0,data.num_nodes)]
    val_mask = [1 if i in val_idx else 0 for i in range(0, data.num_nodes)]

    for epoch in range(1, 20):
        train_loss, train_acc = gcn_train(train_mask)
        val_loss, val_acc = val(val_mask)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train_acc: {train_acc:.4f}, Val_acc: {val_acc:.4f}')
        history['train_loss'].append(train_loss.cpu().detach())
        history['test_loss'].append(val_loss.cpu().detach())
        history['train_acc'].append(train_acc)
        history['test_acc'].append(val_acc)

    foldperf['fold{}'.format(fold+1)] = history
torch.save(model,'./models/k_cross_GCN.pt')

testl_f,tl_f,testa_f,ta_f=[],[],[],[]
for f in range(1,k+1):
     tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
     testl_f.append(np.mean(foldperf['fold{}'.format(f)]['test_loss']))
     ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
     testa_f.append(np.mean(foldperf['fold{}'.format(f)]['test_acc']))
print('==========Model==========')
print(model)
print('Performance of {} fold cross validation'.format(k))
print("Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(np.mean(tl_f),np.mean(testl_f),np.mean(ta_f),np.mean(testa_f)))
'''
import warnings
import datetime
warnings.filterwarnings('ignore')

print(15*'='+'Start Training'+15*'=')
print("Total Epochs:", epochs)

images = []
for epoch in range(1, epochs+1):
    train_loss, train_acc, train_nmi, train_ari = train()
    val_loss, val_acc, val_nmi, val_ari= val()
    print(f'Epoch: {epoch:03d}\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train NMI: {train_nmi:.4f}, Train ARI: {train_ari:.4f}'
          f'\nVal Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val NMI: {val_nmi:.4f}, Val ARI: {val_ari:.4f}')
    with torch.no_grad():
        # save embedding image
        if epoch < 5:
            start = datetime.datetime.now()
            out = model(x, edge_index)
            image = visualize(out, color=y, epoch=epoch)
            images.append(image)

            end = datetime.datetime.now()
            print(15 * '=' + f'TSNE Visualization Image_{epoch} saved. Time: {(end - start).total_seconds():.2f} s' + 15 * '=')

        if epoch % 5 == 0:
            start = datetime.datetime.now()
            out = model(x, edge_index)
            image = visualize(out, color=y, epoch=epoch)
            images.append(image)

            end = datetime.datetime.now()
            print(15 * '=' + f'TSNE Visualization Image_{epoch / 5} saved. Time: {(end-start).total_seconds()} s' + 15 * '=')
images = np.array(images)
np.save(f'./data/gnn/embeddings/images_epoch{epochs}.npy', images)
acc, nmi, ari = test()
print(15*'=' + 'Test Result' + 15*'=')
print(f'test_acc: {acc:.4f}, test_nmi: {nmi:.4f}, test_ari: {ari:.4f}')

torch.save(model,'./models/GAE.pt')
