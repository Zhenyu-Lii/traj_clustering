# Install required packages.
import os
import torch
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
from models import GCNEncoder
from models import GCN
from torch_geometric.nn import VGAE
from torch_geometric.nn import GCNConv

os.environ['TORCH'] = torch.__version__
print(torch.__version__)

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

dataset = torch.load('./data/gnn/processed/data.pt')
data = dataset[0]
out_channels = 2
num_features = data.num_features

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
devices = [torch.device("cuda:" + str(i)) for i in range(4)]
for i in range(len(devices)):
    devices[i] = devices[0]

model.to(devices[0])
x = data.x.to(devices[0])
y = data.y.long().to(devices[1])
edge_index = data.edge_index.to(devices[0])

def gcn_train(train_idx):
    train_mask = torch.tensor(train_idx, dtype=torch.bool)
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    pred = out.argmax(dim=1)  # Use the class with highest probability.
    print(type(pred))
    train_correct = pred[train_mask] == y[train_mask]  # Check against ground-truth labels.
    train_acc = int(train_correct.sum()) / int(train_mask.sum())  # Derive ratio of correct predictions.
    return loss, train_acc

def val(val_idx):
    val_mask = torch.tensor(val_idx, dtype=torch.bool)

    model.eval()
    out = model(x, edge_index)
    loss = criterion(out[train_mask], y[train_mask])
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    correct = pred[val_mask] == y[val_mask]  # Check against ground-truth labels.
    acc = int(correct.sum()) / int(val_mask.sum())  # Derive ratio of correct predictions.
    return loss, acc

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

'''
we calculate the average score in every fold
once the average score is obtained for every fold, we calculate the average score over all the folds.
'''

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

