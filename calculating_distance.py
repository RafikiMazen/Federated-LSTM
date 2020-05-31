from math import cos, asin, sqrt
import os
import pandas as pd
import numpy as np


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a))


os.chdir("./maps")
maps_filenames = os.listdir(".")
maps_indices = [int(x[3:-4])for x in maps_filenames]
maps_indices.sort()
maps = []
for map_index in maps_indices:
    df = pd.read_csv('map' + str(map_index)+".csv", low_memory=False)
    maps.append(df)
os.chdir("../")
for map_filename in maps_filenames:
    cluster_num = int(map_filename[3:-4])
    if not os.path.exists("./Simulation/distances/" + map_filename[3:-4]):
        os.makedirs("./Simulation/distances/" + map_filename[3:-4])
    curr_map = maps[cluster_num]
    distances = pd.DataFrame({}, columns=['closest'])
    for index, row in curr_map.iterrows():
        df = pd.DataFrame({}, columns=['sensor', 'distance'])
        curr_lat = row['latitude']
        curr_long = row['longitude']
        for index1, row1 in curr_map.iterrows():

            row_toAdd = pd.Series({'sensor' : index1,
                                   'distance': distance(row['latitude'], row['longitude'],
                                                        row1['latitude'], row1['longitude'])},
                                  name = index1)
            df = df.append(row_toAdd)
        df = df.sort_values(by = ['distance'])
        path = "./Simulation/distances/" + str(cluster_num) + "/" + str(index) +".csv"
        df.to_csv(path, index=None,header=True)


















