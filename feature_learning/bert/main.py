import time

import numpy as np
import pandas as pd
import argparse
import datetime
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from torch import optim
import pandas as pd
import torch.utils.data as Data
from random import *
from preprocess import DataSet, MyDataSet
from utils import next_batch, get_evalution, make_exchange_matrix, Loss_Function
import config as gl
from utils_1 import gen_train

# init embeddings
loc_index = pd.read_csv('../../data/loc_index.csv',index_col=0)
loc_size = len(loc_index)
embed_size = 256
sgm = 0.8
lbd = 0.6
cwd = os.path.abspath('.')
embed_256 = np.random.randn(loc_size, embed_size)
np.save('data/Geolife/embedding_256.npy', embed_256)

# drop掉label为-1的轨迹
all_traj = pd.read_hdf(cwd+"/all_traj_labeled_σ_{sgm}_λ_{lbd}.h5".format(sgm=sgm, lbd=lbd), key='data')
all_data = all_traj.drop(all_traj[all_traj['label']==-1].index)

# BERT
parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='train device')
parser.add_argument('--bs', default=256, type=int, help='batch size')
parser.add_argument('--epoch', default=10, type=int, help='epoch size')
parser.add_argument('--loss', default='loss', type=str, help='loss fun')
parser.add_argument('--datalen', default=5, type=int, help='datalen')
parser.add_argument('--train_dataset', default="train_traj_5", type=str, help='test dataset')
parser.add_argument('--test_dataset', default="test_traj_5", type=str, help='test dataset')
parser.add_argument('--embed', default='256', type=str, help='loc id embed')
parser.add_argument('--d_model', default=256, type=int, help='embed size')
parser.add_argument('--head', default=2, type=int, help='multi head num')
parser.add_argument('--layer', default=2, type=int, help='layer')

args = parser.parse_args()
device = 'cuda:%s' % args.device
batch_size = args.bs
epoch_size = args.epoch
loss_fun = args.loss

train_dataset = 'all_traj_labeled_σ_{sgm}_λ_{lbd}.h5'.format(sgm=sgm, lbd=lbd)
test_dataset = 'test_traj_' + str(args.datalen) + ".h5"
embed_index = int(args.embed)
print(train_dataset)
print(test_dataset)

# setting invariants
d_model = args.d_model
head = args.head
layer = args.layer
max_pred = args.datalen
gl._init()
gl.set_value('d_model', d_model)
gl.set_value('head', head)
gl.set_value('layer', layer)
gl.set_value('device', device)

# load init embeddings
cwd = os.getcwd()
data_path = os.path.join(os.getcwd(), 'data')

embed_path = os.path.join(data_path, 'Geolife')
if embed_index == 128:
    embed_file = 'embedding_128.npy'
elif embed_index == 256:
    embed_file = 'embedding_256.npy'
elif embed_index == 512:
    embed_file = 'embedding_512.npy'
elif embed_index == 1024:
    embed_file = 'embedding_1024.npy'
else:
    embed_file = "error"

embed_npy = np.load(os.path.join(embed_path, embed_file))
embed_size = embed_npy.shape[1]
gl.set_value('pre_em_size', embed_size)
embed_size = embed_npy.shape[1]
print("embed_file:%s, embed_size: %d" % (embed_file, embed_size))

# load datasets
train_df = pd.read_hdf(os.path.join('./data/Geolife/', train_dataset), key='data')
train_df = train_df.drop(train_df[train_df['label'] == -1].index)
train_data = gen_train(train_df)

# train_word_list1: 20% slower
# time1 = time.time()
# train_word_list1 = list(
#     set(str(train_data[i][0].split(' ')[j]) for i in range(len(train_data)) for j in range(len(train_data[i][0].split(' ')))))
# time2 = time.time()
# delta = time2-time1
train_word_list = list(
    set(word for i in range(len(train_data)) for word in train_data[i][0].split(' ')))
train_word_list.remove('[PAD]')
# time3 = time.time()
# delta1 = time3-time2

# load word list
train_word_list_int = [int(i) for i in train_word_list]

# word2embed 暂时不知道干嘛的
word2embed = dict()
for i, loc in enumerate(train_word_list_int):
    word2embed[str(loc)] = embed_npy[i]
word2embed['[PAD]'] = np.ones(shape=(embed_size,))
word2embed['[MASK]'] = np.ones(shape=(embed_size,))
word2embed['[CLS]'] = np.ones(shape=(embed_size,))
word2embed['[SEP]'] = np.ones(shape=(embed_size,))

word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(train_word_list_int):
    if w == '[PAD]' or w == '[MASK]':
        print("error")
    word2idx[str(w)] = i + 4

idx2word = {i: w for i, w in enumerate(word2idx)}
idx2embed = {i: word2embed[w] for i, w in enumerate(word2idx)}
idx2embed = torch.from_numpy(np.array(list(idx2embed.values())).astype(float))
vocab_size = len(word2idx)
print('vocab_size:', vocab_size)

idx2embed = idx2embed.to(device)

train_token_list = list()
train_user_list = list()
train_day_list = list()
max_value = 0
for sentence in train_data:
    seq, user_index, day = sentence
    seq = seq.split(' ')
    for s in seq:
        max_value = max(max_value, word2idx[s])
    arr = [word2idx[s] for s in seq]
    train_token_list.append(arr)
    train_user_list.append(user_index)
    train_day_list.append(day)

# build exchange_map
start_time = time.time()
print(15*'='+'construct exchange map'+15*'=')
exchange_map = make_exchange_matrix(token_list=train_token_list, token_size=vocab_size)
exchange_map = torch.Tensor(exchange_map).to(device)
end_time = time.time()
print('Time:', end_time-start_time, 's')
print('idx2embed.shape:', idx2embed.shape)
print('vocab_size: ', vocab_size)
print('exchange_map.shape', exchange_map.shape)

exchange_map_copy = exchange_map.clone()
for i in range(len(exchange_map_copy)):
    exchange_map_copy[i][i] = 0

def make_train_data(token_list):
    total_data = []
    for i in range(len(token_list)):
        tokens_a_index = i  # sample random index in sentences
        tokens_a = token_list[tokens_a_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']]

        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]'] and token != word2idx[
                              '[PAD]']]  # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []

        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.7:  # 80%
                input_ids[pos] = word2idx['[MASK]']  # make mask
            elif random() > 0.2:  # 10%
                '''
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                    index = randint(0, vocab_size - 1)
                '''
                co_occ = exchange_map_copy[pos]
                index = co_occ.argmax()
                input_ids[pos] = index  # replace
        total_data.append([input_ids, masked_tokens, masked_pos])
    return total_data

def make_test_data(test_data):
    # [seq, masked_pos, masked_tokens, user_index, day]
    total_test_data = []
    for sentence in test_data:
        arr = [word2idx[s] for s in sentence[0]]
        user = sentence[3]
        arr = [word2idx['[CLS]']] + arr + [word2idx['[SEP]']]
        masked_pos = [pos + 1 for pos in sentence[1]]
        masked_tokens = [word2idx[str(s)] for s in sentence[2]]
        day = sentence[4]
        total_test_data.append([arr, masked_tokens, masked_pos, user, day])
    return total_test_data

total_data = make_train_data(train_token_list)  # [input_ids, masked_tokens, masked_pos]
print("total length of train data is", str(len(total_data)))  #
input_ids, masked_tokens, masked_pos = zip(*total_data)
user_ids, day_ids = torch.LongTensor(train_user_list).to(device), torch.LongTensor(train_day_list).to(device)
input_ids, masked_tokens, masked_pos, = torch.LongTensor(input_ids).to(device), \
                                        torch.LongTensor(masked_tokens).to(device), \
                                        torch.LongTensor(masked_pos).to(device)

# test_total_data = make_test_data(test_data)
# print("total length of test data is", str(len(test_total_data)))

# print(len(total_data), len(test_total_data))

input_ids, masked_tokens, masked_pos = zip(*total_data)
user_ids, day_ids = torch.LongTensor(train_user_list).to(device), torch.LongTensor(train_day_list).to(device)
input_ids, masked_tokens, masked_pos, = torch.LongTensor(input_ids).to(device), \
                                        torch.LongTensor(masked_tokens).to(device), \
                                        torch.LongTensor(masked_pos).to(device)

# test_input_ids, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids = zip(*test_total_data)

loader = Data.DataLoader(MyDataSet(input_ids, masked_tokens, masked_pos, user_ids, day_ids), batch_size, True)

from bert_mlm_model import BERT
model = BERT(vocab_size=vocab_size, id2embed=idx2embed).to(device)
print(15*'='+'Model' + 15*'=')
print(model)
print()

if loss_fun != "loss":
    criterion = Loss_Function()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)
train_predict = []
train_truth = []

# test
def test(test_token_list, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids):
    masked_tokens = np.array(test_masked_tokens).reshape(-1)
    a = list(zip(test_masked_tokens, test_token_list, test_masked_pos, test_user_ids, test_day_ids))
    predict_prob = torch.Tensor([]).to(device)
    MAP = 0
    for batch in next_batch(a, batch_size=64):
        # Value filled with num_loc stands for masked tokens that shouldn't be considered.
        batch_masked_tokens, batch_token_list, batch_masked_pos, batch_user_ids, batch_day_ids = zip(*batch)
        logits_lm = model(torch.LongTensor(batch_token_list).to(device),
                          torch.LongTensor(batch_masked_pos).to(device),
                          torch.LongTensor(batch_user_ids).to(device),
                          torch.LongTensor(batch_day_ids).to(device), )
        logits_lm = torch.topk(logits_lm, 100, dim=2)[1]
        predict_prob = torch.cat([predict_prob, logits_lm], dim=0)

    accuracy_score, fuzzy_score, top3_score, top5_score, top10_score, top30_score, top50_score, top100_score, map_score = \
        get_evalution(ground_truth=masked_tokens, logits_lm=predict_prob, exchange_matrix=exchange_map)

    print('fuzzy score =', '{:.6f}'.format(fuzzy_score))
    print('test top1 score =', '{:.6f}'.format(accuracy_score))
    print('test top3 score =', '{:.6f}'.format(top3_score))
    print('test top5 score =', '{:.6f}'.format(top5_score))
    print('test top10 score =', '{:.6f}'.format(top10_score))
    print('test top30 score =', '{:.6f}'.format(top30_score))
    print('test top50 score =', '{:.6f}'.format(top50_score))
    print('test top100 score =', '{:.6f}'.format(top100_score))
    print('test map score =', '{:.6f}'.format(map_score))

    return 'test accuracy score =' + '{:.6f}'.format(accuracy_score) + '\n' + \
           'fuzzzy score =' + '{:.6f}'.format(fuzzy_score) + '\n' \
           + 'test top3 score =' + '{:.6f}'.format(top3_score) + '\n' \
           + 'test top5 score =' + '{:.6f}'.format(top5_score) + '\n' \
           + 'test top10 score =' + '{:.6f}'.format(top10_score) + '\n' \
           + 'test top30 score =' + '{:.6f}'.format(top30_score) + '\n' \
           + 'test top50 score =' + '{:.6f}'.format(top50_score) + '\n' \
           + 'test top100 score =' + '{:.6f}'.format(top100_score) + '\n' \
           + 'test MAP score =' + '{:.6f}'.format(map_score) + '\n'

# train
for epoch in range(epoch_size):
    train_predict, train_truth = [], []
    for i, (input_ids, masked_tokens, masked_pos, user_ids, day_ids) in enumerate(loader):
        logits_lm = model(input_ids, masked_pos, user_ids, day_ids)
        train_truth.extend(masked_tokens.flatten().cpu().data.numpy())
        train_predict.extend(logits_lm.data.max(2)[1].flatten().cpu().data.numpy())
        if loss_fun == "spatial_loss":
            loss_lm = criterion.Spatial_Loss(exchange_map, logits_lm.view(-1, vocab_size),
                                             masked_tokens.view(-1))  # for masked LM
        else:
            loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))  # for masked LM

        loss = loss_lm


        print('Epoch:', '%06d' % (epoch + 1), 'Iter:', '%06d' % (i + 1), 'loss =', '{:.4f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch%5 ==0:
        # torch.save({'model': model.state_dict()},
        #    'pth_embed/dataset-batch%s-epoch%s-%s-dmodel%s-head%s_layer%s_datalen%s_%s_%s_epoch%d.pth' % (
        #        batch_size, epoch_size, loss_fun, d_model, head, layer, args.datalen, embed_file.replace(".npy", ""),
        #        datetime.datetime.now().strftime("%Y%m%d"),epoch))
        f = open('result2.txt','a+')
        f.write("vocab_size: %d \n" % vocab_size)
        f.write("epoch: %d \n" %epoch)
        f.write("loss: %.6f" % loss)
        f.write('pth_embed/dataset-batch%s-epoch%s-%s-dmodel%s-head%s_layer%s_datalen%s_%s_%s_epoch%d.pth\n' % (
               batch_size, epoch_size, loss_fun, d_model, head, layer, args.datalen, embed_file.replace(".npy", ""),
               datetime.datetime.now().strftime("%Y%m%d"),epoch))
        model.eval()

        # eval every 5 epochs, need test_set
        # result = test(test_input_ids, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids)
        result = 'no test\n'
        model.train()

        f.write(result)
        f.close()

from bert_mlm_model import IntermediateLayerGetter
return_layers = {'linear': 'linear_feature'}
backbone = IntermediateLayerGetter(model, return_layers)
print('===========================Backbone model===============================')
print(backbone)
print('========================================================================')
backbone.eval()
embeddings = torch.zeros(1, 256)
for i, (input_ids, masked_tokens, masked_pos, user_ids, day_ids) in enumerate(loader):
    print(20*'=' + 'i: {}'.format(i) + 20*'=')
    print('No.{}'.format(i))
    print('input_ids: {}; masked_pos: {}; user_id: {}; day_id: {}'.format(input_ids.shape, masked_pos,user_ids,day_ids))
    # print('input_ids: {}; masked_pos: {}; user_id: {}; day_id: {}'.format(input_ids, masked_pos,user_ids,day_ids))
    embedding = backbone(input_ids, masked_pos, user_ids, day_ids)
    # embedding = model(input_ids, masked_pos, user_ids, day_ids, )
    if i == 0:
        embeddings = embedding
    else:
        embeddings = torch.concat((embeddings,embedding),0)

    print(20*'=' + 'Embeddings of batch {}'.format(i) +20*'=')
    print(embeddings)
    print(embeddings.shape)

embed_dic = './data'
torch.save(embeddings, embed_dic + '/Geolife/embed_256.pt')

pth_dic = './pth_model'
torch.save({'model': model.state_dict()},
           pth_dic + '/loc-batch%s-epoch%s-%s-%s.pth' % (
               batch_size, epoch_size, loss_fun,
               datetime.datetime.now().strftime("%Y%m%d%H")))

