# train NN on data
import tensorflow as tf
from tensorflow import keras
 
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Normalization
from keras.layers import LeakyReLU
from keras import optimizers
from keras import regularizers
import numpy as np

filename = "data/datasets/" + "test_data_3_modified.csv"
 
# load the dataset
dataset = loadtxt(filename, delimiter=',', skiprows=1)
# split into input (X) and output (y) variables

X = dataset[:, :-3]
#X = dataset[:, (-1, -2)]

d = X.shape[1]

y = dataset[:,-1]

print(X)
print(y)
opt = keras.optimizers.Adam()

# define the keras model
model = Sequential()
model.add(Normalization())
model.add(Dense(30, 
                input_shape=(X.shape[1],),  
                activation=LeakyReLU(alpha = .1)))
model.add(Dense(30,  
                activation=LeakyReLU(alpha = .1)))


model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
#model.add(Dense(50, activation='relu'))
#model.add(Dense(50, activation='relu'))
#model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.4))
#model.add(Dense(50, activation='relu'))
#model.add(Dense(50, activation='relu'))
#model.add(Dense(50, activation='relu'))
#model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

# compile the keras model
model.compile(loss='huber', optimizer=opt, metrics=['mean_squared_error'])
# fit the keras model on the dataset
model.fit(X, y, epochs=50, batch_size=1, validation_split = .1)
# evaluate the keras model
na, accuracy = model.evaluate(X, y)

#print('Accuracy: %.2f' % accuracy)

output = model.predict(X)
#print(np.sqrt(accuracy))
print("y")
print(y)
print("output")
print(output)

# TODO: have it automatically write to a records file recording parameters and performance for each thing