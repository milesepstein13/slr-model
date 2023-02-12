# train NN on data
import tensorflow as tf
from tensorflow import keras
 
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

filename = "data/datasets/" + "test_data_2_modified.csv"
 
# load the dataset
dataset = loadtxt(filename, delimiter=',', skiprows=1)
# split into input (X) and output (y) variables
X = dataset[:, :-1]
y = dataset[:,-1]
#print(X)
#print(y)
# define the keras model
model = Sequential()
model.add(Dense(30, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# fit the keras model on the dataset
model.fit(X, y, epochs=100, batch_size=50)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)

print('Accuracy: %.2f' % accuracy)

print(np.sqrt(accuracy))

# TODO: have it automatically write to a records file recording parameters and performance for each thing