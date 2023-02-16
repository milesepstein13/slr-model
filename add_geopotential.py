# adds geopotential to the constructed dataset
# modifies dataset to it's the right size/variables to train on
import numpy as np
import pandas as pd
import os
import xarray as xr
import datetime as dt
import scipy

filename = "data/datasets/" + "test_data_2_modified_2.csv"
outfilename = "data/datasets/" + "test_data_2_modified_2_geop.csv"

data = pd.read_csv(filename)

geopotentials = xr.open_dataset('geopotentials.nc')

print(data.shape[0])

data.insert(data.shape[1]-1, "geopotential", np.zeros(data.shape[0]))

lat = data.loc[0, 'latitude']
lon = data.loc[0, 'longitude']
z = geopotentials.interp(latitude = lat, longitude = lon, method='slinear').to_array().values.ravel()
print(lat, lon, z)

for i in range(data.shape[0]):
    
    if not(data.loc[i, 'latitude'] == lat and data.loc[i, 'longitude'] == lon):
        print("new location")
        lat = data.loc[i, 'latitude']
        lon = data.loc[i, 'longitude']
        z = geopotentials.interp(latitude = lat, longitude = lon, method='slinear').to_array().values.ravel()
    data.loc[i, 'geopotential'] = z


print(data)

data.to_csv(outfilename)