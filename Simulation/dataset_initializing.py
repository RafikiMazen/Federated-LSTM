from pandas import read_csv
import pandas as pd
import os

os.chdir("../clustered_sensors/")
os.chdir("../clustered_sensors")
clusters = os.listdir(".")
start_sensor = "0"
numOfRows = 20000
for cluster in clusters:
    df = read_csv("./" + cluster + "/" + start_sensor + ".csv")
    df_b = read_csv("../Simulation/simulation_data/" + cluster +"/" + start_sensor + ".csv")
    df = df[-numOfRows:]
    df_b=df_b[-numOfRows:]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp']=(pd.to_numeric(df.timestamp))/10**11-1
    data_cols = ['f1','f2','timestamp','latitude','longitude','f4']
    target_col=['f3']
    data_cols.append(target_col[0])
    X = pd.DataFrame({},columns=data_cols)
    # X[target_col[0]] = df_b[target_col[0]]
    Y = pd.DataFrame({}, columns=target_col)
    distances_df = read_csv("../Simulation/distances/" + cluster + "/" + start_sensor + ".csv")
    distances_list = list(distances_df["sensor"].values)
    for sensor in distances_list:
        df = read_csv("./" + cluster + "/" + str(int(sensor)) + ".csv")
        df_b = read_csv("../Simulation/simulation_data/" + cluster + "/" + str(int(sensor)) + ".csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # df['timestamp'] = pd.to_numeric(df.timestamp)
        df['timestamp'] = (pd.to_numeric(df.timestamp)) / 10 ** 11
        target_column_in_X = X[target_col[0]]
        X = pd.concat([X[data_cols[:-1]], df[data_cols[:-1]]], axis=0)
        X[target_col[0]] = pd.concat([target_column_in_X, df_b[target_col[0]]], axis=0).values
        Y = pd.concat([Y[target_col], df[target_col]], axis=0)
        print(cluster + "  " + str(len(X)==len(Y)))
    if not os.path.exists("../Simulation/dataset/"+cluster):
        os.makedirs("../Simulation/dataset/"+cluster)
    X.to_csv("../Simulation/dataset/" + cluster + '/X.csv',index=None)
    Y.to_csv("../Simulation/dataset/" + cluster + '/Y.csv', index=None)
