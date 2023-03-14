import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time

from numpy import loadtxt

dataname = 'test_data_6_modified.csv'
filename = "data/datasets/" + dataname
dataset = pd.read_csv(filename)
print(dataset)

lookup = pd.read_csv('nws_lookup')

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
        ret.append(slr_dict[dataset.latitude[i]])
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
        tmax = tempmax(dataset.loc[i])
        if tmax > 271.16:
            pred = 12 + 2*(271.16 - tmax)
        else:
            pred = 12 + (271.16 - tmax)
        ret.append(pred)
    return ret

def predict_nws(dataset):
    ret = []
    depths = dataset.total_precipitation
    temps = dataset['2m_temperature']
    for i in range(dataset.shape[0]):
        SWE_inches = depths[i] * 39.37 * 6
        j = np.argmin(np.abs(lookup[:, 0]-SWE_inches))
        temp = temps[i]
        if temp > 274.261:
            ret.append(0)
        elif temp > 270.928:
            ret.append(lookup[j, 1])
        elif temp > 266.483:
            ret.append(lookup[j, 2])
        elif temp > 263.706:
            ret.append(lookup[j, 3])
        elif temp > 260.928:
            ret.append(lookup[j, 4])
        elif temp > 255.372:
            ret.append(lookup[j, 5])
        elif temp > 244.261:
            ret.append(lookup[j, 6])
        else:
            ret.append(lookup[j, 7])
    return ret

def predict_nn(dataset, num):
    return


# other prediction methods: NWS lookup, ECCC, others from studies
print(predict_kuchera(dataset))
