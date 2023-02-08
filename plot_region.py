import numpy as np
import pandas as pd
import cdsapi
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# This file plots station points and rectangles to help decide where to get data from
# ORDER: NONE

# download stations data
data = pd.read_csv("data/stations_new.csv")   
stations = pd.DataFrame(data)

pc_filestart = "DataSetExport-PC.Working@"
sd_filestart = "DataSetExport-SD.Working@"

lats = []
lons = []


for i, station in stations.iterrows():
    if station['Downloaded']: # do for each usable station
        # find time and location bounaries for data download
        


        # get lat/lon
        lat = station.LATITUDE
        lon = station.LONGITUDE

        lats.append(lat)
        lons.append(lon)

fig, ax = plt.subplots()

ax.plot(lons, lats, 'r*')
left = -125
right = -114.8
bottom = 48.2
top = 51
rect1 = patches.Rectangle((left, bottom), right-left, top-bottom, linewidth=1, edgecolor='r', facecolor='none')

left = -123
right = -118
bottom = 50.75
top = 55.75
rect2 = patches.Rectangle((left, bottom), right-left, top-bottom, linewidth=1, edgecolor='b', facecolor='none')

left = -130.5
right = -123.75
bottom = 51
top = 58
rect3 = patches.Rectangle((left, bottom), right-left, top-bottom, linewidth=1, edgecolor='b', facecolor='none')


ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)

plt.show()