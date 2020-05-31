from fcmeans import FCM
from matplotlib import pyplot as plt
from pandas import DataFrame
import numpy as np
import pandas as pd
from sympy import *
import math
import os

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))
def pattern_generator(y):
    df = pd.DataFrame(y.transpose(), columns=['f1', 'f2', 'f3', 'f4'])
    f_col = [x ** 2 * exp(-0.5 * x) * sin(x + 10) for x in df['f4']]
    df = df.assign(f=f_col).sort_values(by='f').drop('f', axis=1)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    return df


linalg = np.linalg
np.random.seed(8)
numOfRows = 20000
numOfSensors = 200
numOfClusters = 20
start_sensor = 48
x = np.random.normal(size=numOfSensors)
y = np.random.normal(size=numOfSensors)
map = DataFrame(dict(longitude=x, latitude=y, index= range(0, numOfSensors)))
distance_list = []
map.drop('index', inplace=True, axis=1)
for index, row in map.iterrows():
    distances = pd.DataFrame({}, columns=['sensor', 'distance'])
    curr_lat = row['latitude']
    curr_long = row['longitude']
    for index1, row1 in map.iterrows():
        row_toAdd = pd.Series({'sensor': index1,
                               'distance': distance(row['latitude'], row['longitude'],
                                                    row1['latitude'], row1['longitude'])},
                              name=index1)
        distances = distances.append(row_toAdd)
    distances = distances.sort_values(by=['distance'])
    distance_list.append(distances)
plt.scatter(x,y)
fcm = FCM(n_clusters=numOfClusters)
fcm.fit(map[['longitude', 'latitude']])
fcm_centers = fcm.centers
fcm_labels  = fcm.u.argmax(axis=1)
map['cluster_label']=fcm_labels
# plot result
# f, axes = plt.subplots(1, 2, figsize=(11,5))
plt.scatter(map.iloc[:, 0], map.iloc[:, 1])
plt.scatter(map.iloc[:, 0], map.iloc[:, 1], c=fcm_labels)
plt.scatter(fcm_centers.iloc[:,0], fcm_centers.iloc[:,1], marker="s",s=200)
plt.show()
cov =   np.array([[ 1.        ,  0.73886583,  0.654332,  0.56810058],
                 [ 0.73886583,  1.        ,  0.49187904,  0.45994833],
                [ 0.654332,  0.49187904,  1.        ,  0.35644 ],
                [ 0.56810058,  0.45994833,  0.35644 ,  1.        ]])
corr= [[1.000000,   0.502365,     -0.661580],
[   0.502365,   1.000000,     -0.482241],
[-0.661580,  -0.482241,      1.000000]]
sensor_list = []
sensor_df = DataFrame({},columns=['timestamp','f1','f2','f3','f4','longitude','latitude','cluster_label'])
sensor_df['longitude'] = [map['longitude'][start_sensor]] * numOfRows
sensor_df['latitude'] = [map['latitude'][start_sensor]] * numOfRows
init_lat = float(map['latitude'][start_sensor])
init_long = float(map['longitude'][start_sensor])
sensor_df['cluster_label'] = [map['cluster_label'][start_sensor]] * numOfRows
sensor_df['timestamp'] = pd.date_range('2018-01-01 ', periods=numOfRows, freq='30min')
mu = [300,170,90,250]
init_mean_values = mu
correlated = np.random.multivariate_normal(mu, cov, size=numOfRows).transpose()
columns_data = pattern_generator(correlated)
sensor_df['f1'] = columns_data.f1.values
sensor_df['f2'] = columns_data.f2.values
sensor_df['f3'] = columns_data.f3.values
sensor_df['f4'] = columns_data.f4.values
sensor_list.append(sensor_df)
curr_distances = distance_list[start_sensor]
calculated_sensors = [start_sensor]
for i in range(1,numOfSensors):
    print(i)
    for _,row in curr_distances.iterrows():
        if (not(calculated_sensors.__contains__(row.sensor))):
            curr_sensor_index = int(row.sensor)
            distanceToNext = row.distance
            break
    sensor_df = DataFrame({}, columns=['timestamp', 'f1', 'f2', 'f3','f4', 'longitude', 'latitude','cluster_label'])
    sensor_df['longitude'] = [map['longitude'][curr_sensor_index]] * numOfRows
    sensor_df['latitude'] = [map['latitude'][curr_sensor_index]] * numOfRows
    sensor_df['cluster_label'] = [map['cluster_label'][curr_sensor_index]] * numOfRows
    sensor_df['timestamp'] = pd.date_range('2018-01-01 ', periods=numOfRows, freq='30min')
    curr_long = float(map['longitude'][curr_sensor_index])
    curr_lat = float(map['latitude'][curr_sensor_index])
    # mu = [float(x + ((distanceToNext ** (1 / 2)) / 8)) for x in mu]
    mu = [x + float((curr_long-init_long)*11) + float((curr_lat-init_lat)*13) for x in init_mean_values]
    print(curr_sensor_index)
    curr_distances = distance_list[curr_sensor_index]
    correlated = np.random.multivariate_normal(mu, cov, size=numOfRows).transpose()
    columns_data = pattern_generator(correlated)
    print('done generating pattern  ' + str(mu))
    sensor_df['f1'] = columns_data.f1.values
    sensor_df['f2'] = columns_data.f2.values
    sensor_df['f3'] = columns_data.f3.values
    sensor_df['f4'] = columns_data.f4.values
    sensor_list.append(sensor_df)
    calculated_sensors.append(curr_sensor_index)


cluster_list = plot_data = [[] for _ in range(0,numOfClusters)]
for sensor_x in sensor_list:
    cluster_num = sensor_x.cluster_label[1]
    cluster_list[cluster_num].append(sensor_x)

cluster_index = 0
for clustered_sensors in cluster_list:
    sensor_index = 0
    folder_path = './' + 'clustered_sensors/' + str(cluster_index)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for sensor in clustered_sensors:
        sensor.to_csv(folder_path +'/'+ str(sensor_index)+'.csv', header=True, index=range(0,len(sensor)))
        sensor_index+=1
    cluster_index+=1