import pandas as pd
import numpy as np
import pickle
import h5py

'''
把data_cdr.h5转换成E2DTC能跑的格式
'''
# 统计vocab_size, 重新编号
data = pd.read_hdf('./CDR/processed/data_cdr.h5',key='data')
trajs = data['trajectory']
labels = np.array(data['label']).T
srcfile = []
mtafile = []
# 移除pad
for i, traj in enumerate(trajs):
    print(f'Traj {i}')
    traj_list = traj.split()
    traj_list = list(filter(lambda x: x != '[PAD]', traj.split(' ')))
    traj_list = list(map(lambda x: int(x), traj_list))
    # 删掉逗号，以string形式保存
    traj_list = str(traj_list)[1:-1]
    traj_list = list((filter(lambda x: x != ',', traj_list)))
    traj_list= ''.join(traj_list)
    print(traj_list)
    srcfile.append(traj_list)

mtafile = np.array(len(trajs)*[(81.43, 269.48)])
train_length = int(len(trajs)*0.85)

train_src = srcfile[0:train_length]
val_src = srcfile[train_length:]

np.savetxt ('./CDR/processed/train.src', train_src, fmt='%s')
np.savetxt ('./CDR/processed/train.trg', train_src, fmt='%s')
np.savetxt ('./CDR/processed/train.mta', mtafile[0:train_length], fmt='%f')
np.savetxt ('./CDR/processed/val.src', val_src, fmt='%s')
np.savetxt ('./CDR/processed/val.trg', val_src, fmt='%s')
np.savetxt ('./CDR/processed/val.mta', mtafile[train_length:], fmt='%f')
np.savetxt ('./CDR/processed/trj_vocab_cdr.h5', srcfile, fmt='%s')
with open('./CDR/processed/labels_cdr.pkl', 'wb') as file:
    pickle.dump(labels, file)
# srcfile 写成string保存为.h5
# with open('./CDR/processed/train_cdr.src', 'r') as src:
#     vocab_stream = src.readlines()
#     f = h5py.File('./CDR/processed/trj_vocab_cdr.h5', 'w')  # 创建一个h5文件，文件指针是f
#     f['data'] = vocab_stream  # 将数据写入文件的主键data下面
#     f.close()  # 关闭文件
#     # print(vocab_stream)
# #labels 保存为pkl


# with open('./CDR/processed/trj_vocab_cdr.h5', 'r') as file:
#     vocab_stream = file.readlines()
#     print(vocab_stream)

# 计算vocab-dist-cell300.h5
