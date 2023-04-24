import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('data/datasets/test_data_6_modified.csv')
data = data.SLR.values
np.random.seed(0)
np.random.shuffle(data)
n = data.shape[0]
split = int(8*n/10)
train_data = data[:split]
test_data = data[split:]

data = np.round(data, 0)
train_data = np.round(train_data, 0)
test_data = np.round(test_data, 0)

counts = np.zeros([101, 3])

for i in range(101):
    counts[i, 0] = int(np.count_nonzero(data == i))
    counts[i, 1] = int(np.count_nonzero(train_data == i))
    counts[i, 2] = int(np.count_nonzero(test_data == i))


counts = counts.astype(int)
np.savetxt("counts.csv", counts, delimiter=",")