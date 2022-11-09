import pandas as pd
import constants
import numpy as np
'''
import h5py
# 安装h5py库,导入
f = h5py.File('./trj_vocab.h5', 'r')
# 读取文件,一定记得加上路径
for key in f.keys():
    print(f[key].name)
'''

df = pd.read_csv('./loc_index.csv', index_col=0)
df