import os
import glob
import pandas as pd
import numpy as np
import random

if not os.path.exists("./simulation_data/"):
    os.makedirs("./simulation_data/")
os.chdir("../clustered_sensors")
# colnames = ['ozone', 'particullate_matter', 'carbon_monoxide', 'sulfure_dioxide', 'nitrogen_dioxide', 'longitude',
#             'latitude', 'timestamp', 'cluster_label']
filenames = os.listdir(".")
result = []
for filename in filenames:  # loop through all the files and folders
    if os.path.isdir(
            os.path.join(os.path.abspath("."), filename)):
        result.append(filename)
for r in result:
    if not os.path.exists("../Simulation/simulation_data/" + r):
        os.makedirs("../Simulation/simulation_data/" + r)
    path = "./" + r
    extension = 'csv'
    os.chdir(path)
    all_filenames = [c for c in glob.glob('*.{}'.format(extension))]
    counter = 0
    for f in all_filenames:
        df = pd.read_csv(f, low_memory=False)
        length = len(df)
        DESIRED_ROWS = []
        i = 0
        while i <= len(df):
            DESIRED_ROWS.append(i)
            i = i + random.randint(1, 3)
        c = random.randint(1, 3)
        colsToDelete = ['f1','f2','f3','f4']
        z = 1
        while z < c:
            # df.loc[z, 'sensor_status'] = 'ON'
            z += 1
        while c < length:
            if (c in DESIRED_ROWS):
                # df.loc[c,'sensor_status'] = 'OFF'
                for s in colsToDelete:
                    print (str(c) + " " + str(s) + " " + str(len(df)) + " " + str(r) + " " + f)
                    df.loc[c, s] = -1
            else:
                df.loc[c, 'sensor_status'] = 'ON'
            c += 1
        path_l = "../../Simulation/simulation_data/" + r + "/" + f
        df.to_csv(path_l, header=True)
    os.chdir('../')