import torch
import numpy as np
from data_utils import DataOrderScaner, load_label
from data_utils import MyOwnDataset
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
import os

def save_embedding(model, args, cuda0, cuda2):
    autoencoder, clusterlayer = model
    # load init cluster tensor
    '''
    if os.path.isfile(args.cluster_center):
        print("=> Loading cluster center checkpoint '{}'".format(
            args.cluster_center))
        cluster_center = torch.load(args.cluster_center)
        clusters = cluster_center["clusters"]
        n = cluster_center["n_clusters"]
        # load data
        clusterlayer.clusters.data = clusters.to(cuda2)
        return
    '''
    autoencoder.eval()
    clusterlayer.eval()
    vecs = []

    scaner = DataOrderScaner(args.src_file, args.batch)
    scaner.load()  # load trg data

    print("=> Generate representation for all trajectory...")
    while True:
        trjdata = scaner.getbatch()
        if trjdata is None:
            break
        src, lengths, invp = trjdata.src, trjdata.lengths, trjdata.invp
        src, lengths = src.to(cuda0), lengths.to(cuda0)
        # (batch, hidden_size * num_directions)
        context = autoencoder.encoder_hn(src, lengths)
        context = context[invp]
        vecs.append(context.cpu().data)

    # 在这里生成了所有的embedding，可以构造全局的KNNG，考虑使用rNNG，怎么定义radius呢？
    vecs = torch.cat(vecs)

    X = np.array(vecs.cpu())
    #改成radius
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    A = nbrs.kneighbors_graph(X).toarray()
    Y = load_label(args.label_file)

    edge_index, _ = from_scipy_sparse_matrix(coo_matrix(A))
    print("-" * 7 + "Dataset done!" + "-" * 7)
    print("-" * 7 + "edge index!" + "-" * 7)
    print(edge_index)
    '''
    构建GNN的数据集
    '''
    # 放入datalist
    dataset = MyOwnDataset('./data/gnn', edge_index, X, Y)

'''
    torch.save({
        "clusters": clusterlayer.clusters.data.cpu(),
        "n_clusters": args.n_clusters
    }, args.cluster_center)
    print("-" * 7 + "Initiated cluster center" + "-" * 7)
'''

def target_distribution(q):
    # clustering target distributio
    # \n for self-training
    # q (batch,n_clusters): similarity between embedded point and cluster center
    # p (batch,n_clusters): target distribution
    weight = q**2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t()
    return p 


def update_cluster(dtc, args, cuda0, cuda2):
    q = []
    vecs = []
    dtc.eval()
    autoencoder, clusterlayer = dtc.autoencoder, dtc.clusterlayer

    scaner = DataOrderScaner(args.src_file, args.batch)
    scaner.load()  # load trg data
    while True:
        trjdata = scaner.getbatch()
        if trjdata is None:
            break
        src, lengths, invp = trjdata.src, trjdata.lengths, trjdata.invp
        src, lengths = src.to(cuda0), lengths.to(cuda0)

        # get trajectory represention
        # (batch, hidden_size * num_directions)
        context = autoencoder.encoder_hn(src, lengths)
        context = context[invp]
        '''
        use this context to construct KNN Graph
        Input: context(features) 
    
        X = np.array(context.cpu())
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        A = nbrs.kneighbors_graph(X).toarray()
        '''

        # q_i (batch,n_clusters)
        q_i = clusterlayer(context)
        q.append(q_i.cpu().data)
        vecs.append(context.cpu().data)

    # (datasize,n_clusters)
    q = torch.cat(q)
    vecs = torch.cat(vecs)

    dtc.train()
    return vecs, q, target_distribution(q)