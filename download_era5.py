import numpy as np
import pandas as pd
# download stations data
data = pd.read_csv("data/stations.csv")   
stations = pd.DataFrame(data)


# for each station
for station in stations.iterrows():
    print(station)

    # download its data

    # put its start and endtime in stations data
    # download era5 data w/ appropriate variables at appropriate time and location 
    # save era5 data as csv