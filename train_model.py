# train NN on data
import tensorflow as tf
from tensorflow import keras
 
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np

filename = "data/datasets/" + "test_data_2_modified.csv"
 
# load the dataset
dataset = loadtxt(filename, delimiter=',', skiprows=1)
# split into input (X) and output (y) variables
X = dataset[:, :-1]
X = X[:, 52:55]
y = dataset[:,-1]
print(X.shape)
#print(y)
# define the keras model
model = Sequential()
model.add(Dropout(0.4))
model.add(Dense(50, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='linear'))

# compile the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
# fit the keras model on the dataset
model.fit(X, y, epochs=10, batch_size=50)
# evaluate the keras model
output, accuracy = model.evaluate(X, y)

print('Accuracy: %.2f' % accuracy)

print(np.sqrt(accuracy))

# TODO: have it automatically write to a records file recording parameters and performance for each thing