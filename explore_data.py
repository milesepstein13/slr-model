import matplotlib.pyplot as plt

import numpy as np
from numpy import loadtxt

filename = "data/datasets/" + "test_data_4_modified.csv"
 
# load the dataset
dataset = loadtxt(filename, delimiter=',', skiprows=1)
# split into input (X) and output (y) variables

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

plt.scatter(out[:, 1], out[:, 0], c = out[:, 2], s=out[:, 3]/2, cmap='rainbow')

plt.title("Mean SLR by location")
plt.colorbar()
plt.xlabel("lon")
plt.ylabel("lat")
plt.savefig('SLR_locations.png')
plt.clf()


plt.scatter(out[:, 4], out[:, 5], c = out[:, 2], s=out[:, 3]/2, cmap='rainbow')

plt.title("Mean SLR by Northeasternness and elevation")
plt.colorbar()
plt.xlabel("northeasternness (lat-lon)")
plt.ylabel("elevation")
plt.savefig('SLR_NE_elev.png')
plt.clf()
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