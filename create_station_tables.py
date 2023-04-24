import numpy as np
import pandas as pd


stations = pd.read_csv('data/stations_new.csv')

print(stations)

stations = stations[stations.Downloaded == True]

stations_list = stations.drop(columns = ['SNWASWS_ID', 'STATUS', 'Unnamed: 0'])
stations_list = stations_list.iloc[:, :5]


print(stations_list.LATITUDE)

clusters = np.array([])
for i in range(len(stations_list.LATITUDE)):
    lat = stations_list.LATITUDE.iloc[i]
    lon = stations_list.LONGITUDE.iloc[i]
    if (lon < -122.5 and lat < 54):
        clusters = np.append(clusters, ['Coast Range'])
    elif (lon < -122.5 ):
        clusters = np.append(clusters, ['Rockies'])
    else:
        clusters = np.append(clusters, ['Northern BC'])

stations_list['Region'] = clusters

stations_errors = stations_list

stations_list = stations_list.round(3)

stations_errors = stations_errors.drop(columns=['ELEVATION', "LONGITUDE"])

data = pd.read_csv('data/datasets/test_data_6_modified.csv')
val = data['latitude']
values = val.values
#print(values)
np.random.seed(0)
np.random.shuffle(values)
split = n = values.shape[0]
split = int(8*n/10)
lats = values[split+1:]
lats = np.sort(np.unique(lats))

mae = pd.read_csv('outputs/station_mae.csv')
rmse = pd.read_csv('outputs/station_rmse.csv')

stations_mae = stations_errors

for i in range(21):
    stations_mae[str(i+1)] = 0

stations_mae['Count'] = 0


for i in range(lats.shape[0]):
    lat = lats[i]
    row = list(stations_mae.LATITUDE.values).index(lat)
    stations_mae.iloc[row, 4:] = mae.iloc[i, 2:]

stations_mae = stations_mae.round(2)



stations_rmse = stations_errors

for i in range(21):
    stations_rmse[str(i+1)] = 0

stations_rmse['Count'] = 0


for i in range(lats.shape[0]):
    lat = lats[i]
    row = list(stations_rmse.LATITUDE.values).index(lat)
    stations_rmse.iloc[row, 4:] = rmse.iloc[i, 2:]

stations_rmse = stations_rmse.round(2)

stations_mae = stations_mae.drop(columns=['LATITUDE'])
stations_rmse = stations_rmse.drop(columns=['LATITUDE'])

print(stations_rmse)
print(stations_mae)
print(stations_list)

stations_rmse.to_csv('outputs/stations_rmse_neat.csv')
stations_mae.to_csv('outputs/stations_mae_neat.csv')
stations_list.to_csv('outputs/stations_list.csv')