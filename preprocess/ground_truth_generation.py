import os
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from scipy.spatial.distance import pdist, cdist
import time

start_time = time.time()
def geo_distance(c_i, c_j):
    lng1, lat1 = c_i[0], c_i[1]
    lng2, lat2 = c_j[0], c_j[1]
    #lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance,3)
    # print(c_i, c_j, distance)
    return distance

def count_points(t):
    # traj = t[0].split(' ') # Geolife
    traj = t[0] # CDR
    pad_cnt = traj.count('[PAD]')
    cnt = len(traj) - pad_cnt
    return cnt

def rangeQuery(t, c, c_number, dm):
    # 补一个token->(lon,lat)的逻辑
    cnt = 0
    # coo_list = []
    dist_list = []
    dist_list_1 = []
    # traj = t[0].split(' ')# geolife
    traj = t[0] # CDR
    # print('traj:', traj)
    for i in range(len(traj)):
        if traj[i] == '[PAD]':
            continue
        token = int(traj[i])
        # print(token)
        # coo = tuple((loc_index.iloc[token][0], loc_index.iloc[token][1]))
        # distance_1 = dm.values[token][c_number] # Geolife
        distance_1 = dm.loc[token][c_number] # CDR
        # distance = geo_distance(coo, (c[0], c[1]))
        if distance_1 < c[2]:
            cnt += 1
        # coo_list.append((loc_index.iloc[token][0], loc_index.iloc[token][1]))
        # dist_list.append(distance)
        dist_list_1.append(distance_1)

    # for coo in coo_list:
    #     distance = geo_distance(coo, (c[0], c[1]))
    #     dist_list_2.append(distance)
    #     # if distance < c[2]:
    #     #     cnt += 1

    # 用于检查两种distance计算方法的结果一不一样
    # for i in range(len(dist_list_1)):
    #     print(dist_list[i], dist_list_1[i])
    return cnt

dataset_name = 'CDR'
# dataset_name = 'Geolife'

# CDR
loc_index = pd.read_csv('./CDR/jinchengFilteredLoc.csv',index_col='loc_index')[['lng','lat']]
cluster_center = pd.read_csv('./CDR/processed/cluster_center_15.csv',index_col=0)[['longitude','latitude']]
cluster_center.columns = ['lon','lat']
all_traj = pd.read_hdf('./CDR/train_traj_5.h5', key='data')
sgm = 0.8
lbd = 0.6
cluster_num = 20

radius = pdist(cluster_center.values.tolist(), lambda c_i, c_j: geo_distance(c_i, c_j)).min()
cluster_center['radius'] = radius * sgm
locs = loc_index.values.tolist()
centers = cluster_center[['lon','lat']].values.tolist()

# 保存每个点（loc_index）到每个质心的距离
dm = cdist(locs, centers, lambda loc, center: geo_distance(loc, center))
dm = pd.DataFrame(dm, columns=[('C_'+str(i))for i in range(len(cluster_center))])
dm.index = loc_index.index
dm.index.name = 'loc_index'
dm.to_hdf(f'./{dataset_name}/processed/distance_matrix_{cluster_num}.h5', key='data')

dm = pd.read_hdf(f'./{dataset_name}/processed/distance_matrix_{cluster_num}.h5', key='data')


label = len(all_traj)*[-1]
fallen_table = []
# go through every traj
for j, t in enumerate(all_traj.values):
    cnt_list = []
    fallen_rate_list = []
    points_num = count_points(t)
    # calculate fallenRate to each cluster
    for i, c in enumerate(cluster_center.values):
        fallenPoints = rangeQuery(t, c, i, dm)
        fallenRate = fallenPoints/points_num
        fallen_rate_list.append((fallenRate,i))
        cnt_list.append(fallenPoints)
        # break
    cnt_list.append(points_num)
    fallen_table.append(cnt_list)
    fallen_rate_list.sort(key=lambda x: x[0], reverse=True)

    # choose cluster with max fallenRate
    # label[j] = fallen_rate_list[0][1]

    # use lambda to determine cluster
    label[j] = fallen_rate_list[0][1] if fallen_rate_list[0][0]>lbd else -1
    print("Traj.No:", j, 'Max fallenRate: %.2f'%fallen_rate_list[0][0], 'Label:', label[j])
    print(cnt_list)
    print(fallen_rate_list)
    print()

df_fallen_table = pd.DataFrame(fallen_table, columns=[('C_'+str(i))for i in range(len(cluster_center))]+['points_num'])
df_fallen_table['sum'] = df_fallen_table.iloc[:,:-1].sum(axis=1)
df_fallen_table['points_no_label'] = df_fallen_table['points_num'] - df_fallen_table['sum']
df_fallen_table.index.name = 'traj_no'
df_fallen_table.to_hdf('./fallen_table.h5', key='data')
#%%
df_fallen_table = pd.read_hdf('./fallen_table.h5', key='data')
# points_sum: 轨迹有多少个points
# sum: 轨迹中有多少个点落在这12个cluster内
# sum == 0 则表示整条轨迹没有任何一个点可以被分到这12类里
#%%
print(df_fallen_table['sum'].value_counts())

all_traj['label'] = label
all_traj.to_hdf(f'./CDR/processed/all_traj_labeled_σ_{sgm}_λ_{lbd}_{cluster_num}.h5', key='data')
end_time = time.time()
print(f'Time: {start_time-end_time:.2f}s')