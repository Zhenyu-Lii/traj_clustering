import pandas as pd
import numpy as np
import pickle

'''
把data_cdr.h5转换成E2DTC能跑的格式
'''

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
    print(traj_list)
    srcfile.append(traj_list)

mtafile = np.array(len(trajs)*[(81.43, 269.48)])

np.savetxt ('./CDR/processed/train_cdr.src', srcfile, fmt='%s')
np.savetxt ('./CDR/processed/train_cdr.trg', srcfile, fmt='%s')
np.savetxt ('./CDR/processed/train_cdr.mta', mtafile, fmt='%f')
#labels 保存为pkl
with open('./CDR/processed/labels_cdr.pkl', 'wb') as file:
    pickle.dump(labels, file)