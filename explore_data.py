import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
from numpy import loadtxt
import pandas as pd
from scipy import stats

filename = "data/datasets/" + "test_data_4.csv"
 
# load the dataset

# TODO: make scale of axes same for all plots, fix bin width artifact
dataset = pd.read_csv(filename)
print(dataset)

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

plt.hist2d(dataset.pc_increase, dataset.sd_increase, norm=mpl.colors.LogNorm(), bins=50)
plt.xlabel("Cumulative Precip Increase (mm)")
plt.ylabel("Snow Depth Increase (cm)")
plt.title('PC and SD increases for whole dataset')
plt.colorbar()
plt.savefig('pc_sd_plots/pc_sd_density.png')
plt.clf()

dataset = dataset.assign(cluster = get_clusters(dataset), station = get_station_nums(dataset))
print(dataset)

plt.hist2d(dataset[dataset.cluster == 1].pc_increase, dataset[dataset.cluster == 1].sd_increase, norm=mpl.colors.LogNorm(), bins=50)
plt.xlabel("Cumulative Precip Increase (mm)")
plt.ylabel("Snow Depth Increase (cm)")
plt.title('PC and SD increases for coast range')
plt.colorbar()
plt.savefig('pc_sd_plots/pc_sd_density_coast.png')
plt.clf()

plt.hist2d(dataset[dataset.cluster == 2].pc_increase, dataset[dataset.cluster == 2].sd_increase, norm=mpl.colors.LogNorm(), bins=50)
plt.xlabel("Cumulative Precip Increase (mm)")
plt.ylabel("Snow Depth Increase (cm)")
plt.title('PC and SD increases for interior')
plt.colorbar()
plt.savefig('pc_sd_plots/pc_sd_density_interior.png')
plt.clf()

for i in range(1, 66):
    print("i: " + str(i))
    plt.hist2d(dataset[dataset.station == i].pc_increase, dataset[dataset.station == i].sd_increase, norm=mpl.colors.LogNorm(), bins=20)
    plt.xlabel("Cumulative Precip Increase (mm)")
    plt.ylabel("Snow Depth Increase (cm)")
    plt.title('PC and SD increases for station ' + str(i))
    plt.colorbar()
    plt.savefig('pc_sd_plots/pc_sd_density_station_' + str(i) + '.png')
    plt.clf()

print(np.max(dataset.station))
print(np.min(dataset.station))

# split into input (X) and output (y) variables
if False:
    slrs = dataset[:, -1]
    lats = dataset[:, -4]
    lons = dataset[:, -3]
    elevs = dataset[:, -5]

    lat = 0
    lon = 0
    count = 0
    sum = 0
    out = [0, 0, 0, 0, 0, 0]

    for i in range(len(slrs)):
        if ((lats[i] != lat) ):
            print("new location " + str(i))
            if lat != 0:
                slr = sum/count
                out = np.vstack([out, [lat, lon, slr, count, lat-lon, elev]])
                count = 0
                sum = 0
            lat = lats[i]
            lon = lons[i]
            elev = elevs[i]
        sum = sum + slrs[i]
        count = count + 1
    out = out[1:, :]
    print(out)

#plt.scatter(out[:, 1], out[:, 0], c = out[:, 2], s=out[:, 3]/2, cmap='rainbow')

#plt.title("Mean SLR by location")
#plt.colorbar()
#plt.xlabel("lon")
#plt.ylabel("lat")
#plt.savefig('SLR_locations.png')
#plt.clf()


#plt.scatter(out[:, 4], out[:, 5], c = out[:, 2], s=out[:, 3]/2, cmap='rainbow')

#plt.title("Mean SLR by Northeasternness and elevation")
#plt.colorbar()
#plt.xlabel("northeasternness (lat-lon)")
#plt.ylabel("elevation")
#plt.savefig('SLR_NE_elev.png')
#plt.clf()
#plt.hist(dataset[:, -1], bins=20)
#plt.title("SLR Histogram")
#plt.xlabel("SLR")
#plt.savefig('SLR.png')
#plt.clf()

#plt.hist(dataset[:, -3], bins=20)
#plt.title("Cumulative Precipitation Increase Histogram")
#plt.xlabel("Cumulative Precipitation Increase (mm)")
#plt.savefig('PC.png')
#plt.clf()

#plt.hist(dataset[:, -2], bins=20)
#plt.title("Snow Depth Increase Histogram")
#plt.xlabel("Snow Depth Increase (cm)")
#plt.savefig('SD.png')