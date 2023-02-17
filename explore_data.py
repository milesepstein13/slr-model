import matplotlib.pyplot as plt

import numpy as np
from numpy import loadtxt

filename = "data/datasets/" + "test_data_3_modified.csv"
 
# load the dataset
dataset = loadtxt(filename, delimiter=',', skiprows=1)
# split into input (X) and output (y) variables

plt.hist(dataset[:, -1], bins=20)
plt.title("SLR Histogram")
plt.xlabel("SLR")
plt.savefig('SLR.png')
plt.clf()

plt.hist(dataset[:, -3], bins=20)
plt.title("Cumulative Precipitation Increase Histogram")
plt.xlabel("Cumulative Precipitation Increase (mm)")
plt.savefig('PC.png')
plt.clf()

plt.hist(dataset[:, -2], bins=20)
plt.title("Snow Depth Increase Histogram")
plt.xlabel("Snow Depth Increase (cm)")
plt.savefig('SD.png')