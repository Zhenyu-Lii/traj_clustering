Geolife数据处理方法

##### Attenmove

we filter out the trajectories with less than 12 time slots and the users with less than 5 day’s trajectories for Geolife.

![image-20220923145931554](C:/Users/lenovo/AppData/Roaming/Typora/typora-user-images/image-20220923145931554.png)

##### our method（E2DTC）

| Dataset |  City   | Duration | #Users | #Traj | #Distinctive Loc（all） |
| :-----: | :-----: | :------: | :----: | :---: | :---------------------: |
| Geolife | Beijing | 5 years  |   66   | 1731  |          68589          |
|         |         |          |        |       |                         |

- 所有的cluster center都没有出现在geolife轨迹中

##### 文件

fallen_table.h5：每条轨迹落在各个聚类中心的采样点的数目

distance_matrix.h5：每个token到各个聚类中心的距离矩阵

all_traj_timeslotId.h5：带有timeslot标签的all_traj

loc_index.csv：每个token对应的经纬度

