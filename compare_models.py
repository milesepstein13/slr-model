import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time
from numpy import loadtxt
import tensorflow as tf
from tensorflow import keras

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
    split = int(9*n/10)
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


predictions = pd.DataFrame().assign(recorded = validation_datasets[""].SLR, 
                                    ten = predict_ten(validation_datasets[""]), 
                                    meann = predict_mean(validation_datasets[""]),
                                    station_mean = predict_station_mean(validation_datasets[""]),
                                    kuchera = predict_kuchera(validation_datasets[""]),
                                    nn = predict_nn(validation_datasets[""], 174))

print(predictions)


SE = pd.DataFrame().assign(recorded = ((predictions.recorded-predictions.recorded)**2), 
                            ten = ((predictions.recorded-predictions.ten)**2), 
                            mean = ((predictions.recorded-predictions.meann)**2),
                            station_mean = ((predictions.recorded-predictions.station_mean)**2),
                            kuchera = ((predictions.recorded-predictions.kuchera)**2),
                            nn = ((predictions.recorded-predictions.nn)**2))

MSE = SE.mean()

print(MSE)

# make figures
# compare perforamce at different locations