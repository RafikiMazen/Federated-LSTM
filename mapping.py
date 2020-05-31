import os
import pandas as pd
import glob

os.chdir(".\clustered_sensors")
rootdir = '.\clustered_sensors'
colnames= ['ozone','particullate_matter','carbon_monoxide','sulfure_dioxide','nitrogen_dioxide','longitude','latitude','timestamp','cluster_label']
filenames = os.listdir(".")  # get all files' and folders' names in the current directory
result = []
data = {}
for filename in filenames:  # loop through all the files and folders
    if os.path.isdir(
            os.path.join(os.path.abspath("."), filename)):  # check whether the current object is a folder or not
        result.append(filename)
for r in result:
    result_df = pd.DataFrame(data, columns=['longitude', 'latitude'])
    path = "./" + str(r)
    extension = 'csv'
    os.chdir(path)
    all_filenames = [c for c in glob.glob('*.{}'.format(extension))]
    counter = 0
    for f in all_filenames:
        df = pd.read_csv(f, low_memory=False)
        row = pd.Series({'latitude':df.loc[1,'latitude'], 'longitude':df.loc[1,'longitude']}, name = int(f[:-4]))
        counter += 1
        result_df = result_df.append(row)
    result_df.sort_index()
    longitude_array = result_df.longitude.unique()
    longitude_array.sort()
    latitude_array = result_df.latitude.unique()
    latitude_array.sort()
    graph = pd.DataFrame(data, columns = longitude_array, index=latitude_array)
    # if (len(latitude_array)!= len(longitude_array)):
    #     raise Exception("multiple sesnors on one axis")
    for long in longitude_array:
        for lat in latitude_array:
            m = result_df.loc[
                (result_df['longitude'] == long) &
                (result_df['latitude'] == lat)]
            graph.loc[lat, long] = list(m.index.values)
            # row = pd.Series({long: list(m.index.values)}, name=lat)
            # graph = graph.append(row)
    path_m = "../../maps/" + "map" + r + ".csv"
    path_g = "../../maps/" + "graph" + r + ".csv"
    if not os.path.exists("../../maps"):
            os.makedirs("../../maps")
    result_df = result_df.sort_index()
    result_df.to_csv(path_m, index=True, header=True)
    os.chdir('../')