import numpy as np
import pandas as pd
import os

# download stations data
data = pd.read_csv("data/stations_new.csv")   
stations = pd.DataFrame(data)

pc_filestart = "DataSetExport-PC.Working@"
sd_filestart = "DataSetExport-SD.Working@"

# for each station
for i, station in stations.iterrows():
    if station['Downloaded']: # do for each usable station
        print(i)
        # download era5 data w/ appropriate variables at appropriate time and location 

        # save era5 data as csv.