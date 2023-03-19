import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras
import scipy as sp
from scipy import stats

np.random.seed(0)

# to get only validation examples from dataset
def get_validation(dataset):
    values = dataset.values
    np.random.shuffle(values)
    print(values.shape)
    print(dataset.shape)
    dataset = pd.DataFrame(values[:, 1:], columns =dataset.columns[1:])
    print(dataset)
    n = dataset.shape[0]
    split = int(8*n/10)
    dataset = dataset.iloc[split + 1:, :]
    return dataset

base_name = 'data/datasets/test_data_6_modified'

# load in validation datasets (each dataset contains different variables)
dataset_names = ["", "_small", "_without_derived", "_without_meta", "_without_radiation", "_without_surface", "_without_temperature", "_without_upper", "_without_water", "_without_wind"]
validation_datasets = dict()
for name in dataset_names:
    validation_datasets[name] = get_validation(pd.read_csv(base_name + name + '.csv'))


lookup = pd.read_csv('nws_lookup.csv')

def predict_ten(dataset):
    return 10*np.ones(dataset.shape[0])

def predict_mean(dataset):
    return np.ones(dataset.shape[0]) * np.mean(dataset.SLR)

def predict_station_mean(dataset):
    # distinguishes stations by latitude (assumes two dont have same lat)
    lats = list(set(dataset.latitude))
    slrs = np.zeros(len(lats))
    for i in range(len(lats)):
        slrs[i] = np.mean(dataset.SLR[dataset.latitude == lats[i]])
    slr_dict = dict(zip(lats, slrs))
    ret = []
    for i in range(dataset.shape[0]):
        ret.append(slr_dict[dataset.latitude.iloc[i]])
    return ret

def tempmax(row):
    # returns max temp in lowest 500 mb
    max = row['2m_temperature']
    surface_pressure = row.surface_pressure/100
    pressure_levels = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200]
    tempstr = 'temperature_'
    for pressure in pressure_levels:
        if (pressure + 500 > surface_pressure) and (pressure < surface_pressure):
            st = tempstr + str(pressure)
            if row[st] > max:
                max = row[st]
    return max

def predict_kuchera(dataset):
    ret = []
    for i in range(dataset.shape[0]):
        tmax = tempmax(dataset.iloc[i])
        if tmax > 271.16:
            pred = 12 + 2*(271.16 - tmax)
        else:
            pred = 12 + (271.16 - tmax)
        ret.append(pred)
    return ret

# only a lookup table, not used operationally
def predict_nws(dataset):
    ret = []
    depths = dataset.total_precipitation
    temps = dataset['2m_temperature']
    for i in range(dataset.shape[0]):
        SWE_inches = depths[i] * 39.37 * 6
        if SWE_inches < .008:
            SWE_inches = .01
        j = np.argmin(np.abs(lookup.SWE-SWE_inches))
        temp = temps[i]
        if temp > 274.261:
            ret.append(0)
        elif temp > 270.928:
            ret.append(lookup.loc[j]['28-34']/SWE_inches)
        elif temp > 266.483:
            ret.append(lookup.loc[j]['20-27']/SWE_inches)
        elif temp > 263.706:
            ret.append(lookup.loc[j]['15-19']/SWE_inches)
        elif temp > 260.928:
            ret.append(lookup.loc[j]['10-14']/SWE_inches)
        elif temp > 255.372:
            ret.append(lookup.loc[j]['0-9']/SWE_inches)
        elif temp > 244.261:
            ret.append(lookup.loc[j]['-1--20']/SWE_inches)
        else:
            ret.append(lookup.loc[j]['-21--40']/SWE_inches)
    return ret

def predict_nn(dataset, num):
    model = keras.models.load_model("models/"+str(num))
    print(dataset)
    X = dataset.to_numpy()[:, :-1]
    print(X)
    return(model.predict(X))

# other prediction methods: ECCC, others from studies

def get_station_nums(dataset):
    return stats.rankdata(dataset.latitude, method='dense')

def get_clusters(dataset):
    clusters = np.zeros(len(dataset.latitude))
    for i in range(len(dataset.latitude)):
        lat = dataset.latitude.iloc[i]
        lon = dataset.longitude.iloc[i]
        if (lon < -127 or (lon < -122.5 and lat < 54.5) or (lon < -120 and lat < 51)):
            clusters[i] = 1
        else:
            clusters[i] = 2
    plt.scatter(dataset.longitude, dataset.latitude, c = clusters, cmap='rainbow')

    plt.title("Regions")
    plt.colorbar()
    plt.xlabel("lon")
    plt.ylabel("lat")
    plt.savefig('regions.png')
    plt.clf()
    return clusters

predictions = pd.DataFrame().assign(recorded = validation_datasets[""].SLR, 
                                    ten = predict_ten(validation_datasets[""]), 
                                    meann = predict_mean(validation_datasets[""]),
                                    station_mean = predict_station_mean(validation_datasets[""]),
                                    kuchera = predict_kuchera(validation_datasets[""]),
                                    nn = predict_nn(validation_datasets[""], 174),
                                    station = get_station_nums(validation_datasets[""]),
                                    cluster = get_clusters(validation_datasets[""]))

print(predictions)


SE = pd.DataFrame().assign(recorded = ((predictions.recorded-predictions.recorded)**2), 
                            ten = ((predictions.recorded-predictions.ten)**2), 
                            mean = ((predictions.recorded-predictions.meann)**2),
                            station_mean = ((predictions.recorded-predictions.station_mean)**2),
                            kuchera = ((predictions.recorded-predictions.kuchera)**2),
                            nn = ((predictions.recorded-predictions.nn)**2),
                            station = predictions.station,
                            cluster = predictions.cluster)
print(SE)

MSE = SE.mean()
print("Overall:")
print(MSE)


SE_stations = SE.groupby('station')
MSE_locations = SE_stations.mean()
MSE_locations['count'] = SE_stations.count().recorded
print("Locations:")
print(MSE_locations)


SE_clusters = SE.groupby('cluster')
MSE_clusters = SE_clusters.mean()
MSE_clusters['count'] = SE_clusters.count().recorded
print("Clusters:")
print(MSE_clusters)


# save data
predictions.to_csv('outputs/predictions.csv')
MSE.to_csv('outputs/overall_mse.csv')
MSE_locations.to_csv('outputs/station_mse.csv')
MSE_clusters.to_csv('outputs/cluster_mse.csv')

# put tables to latex:
print(MSE.to_latex())
print(MSE_locations.to_latex())
print(MSE_clusters.to_latex())

# make figures
# not sure what figures to make? Should put station-by-station data on a map. Ask rosie what

def make_maps(MSE_locations, lats, lons):
    for column in MSE_locations:
        if (column != 'recorded') and (column != 'cluster') and (column != 'count'):
            plt.scatter(lons, lats, c = MSE_locations[column], s=MSE_locations['count'], cmap='rainbow')

            plt.title("Mean Test Squared Error With " + column + " Method for SLR Prediction")
            plt.colorbar()
            plt.xlabel("lon")
            plt.ylabel("lat")
            plt.savefig('outputs/figs/' + column + '.png')
            plt.clf()


def get_lat_lon(dataset):
    lats = np.unique(dataset.latitude)
    lats_order = list(dict.fromkeys(dataset.latitude))
    lons_order = list(dict.fromkeys(dataset.longitude))
    dic = dict(zip(lats_order, lons_order))
    lons = []
    for lat in lats:
        lons = lons + [dic[lat]]
    return (lats, lons)


(lats, lons) = get_lat_lon(validation_datasets[''])
make_maps(MSE_locations, lats, lons)