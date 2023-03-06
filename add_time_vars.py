# adds geopotential to the constructed dataset
# modifies dataset to it's the right size/variables to train on
import numpy as np
import pandas as pd
import os
import xarray as xr
import datetime as dt
import scipy
from dateutil import parser

filename = "data/datasets/" + "test_data_4_modified.csv"
outfilename = "data/datasets/" + "test_data_5_modified.csv"

data = pd.read_csv(filename)


print(data.shape[0])

data.insert(data.shape[1]-2, "doy", np.zeros(data.shape[0]))
data.insert(data.shape[1]-3, "time", np.zeros(data.shape[0]))


for i in range(data.shape[0]):
    
    time = data.loc[i, 'starttime']
    time = parser.parse(time)
    #print(time)
    #print(time.timetuple().tm_yday)
    #print(time.hour)

    data.loc[i, 'doy'] = time.timetuple().tm_yday
    data.loc[i, 'time'] = time.hour + 3
print(data)

data.to_csv(outfilename)