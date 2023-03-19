# trains all final models to be compared leaving data for validation

# train NN on data
import tensorflow as tf
from tensorflow import keras
 
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Normalization
from keras.layers import LeakyReLU
from keras.layers import ReLU
from keras import optimizers
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import time

np.random.seed(0)


datanames = ['test_data_6_modified.csv', 
             'test_data_6_modified_small.csv', 
             'test_data_6_modified_without_derived.csv', 
             'test_data_6_modified_without_meta.csv', 
             'test_data_6_modified_without_radiation.csv', 
             'test_data_6_modified_without_surface.csv', 
             'test_data_6_modified_without_temperature.csv', 
             'test_data_6_modified_without_upper.csv', 
             'test_data_6_modified_without_water.csv', 
             'test_data_6_modified_without_wind.csv']

num_models = len(datanames)
print(num_models)

count = 0
for dataname in datanames:
    count = count+1
    # open saved model info
    models = pd.read_csv('models.csv')
    model_id = models.shape[0]
    print(str(model_id))
    print(str(count) + ' out of ' + str(num_models))
    #dataname = "test_data_4_modified_small.csv"
    optimizer = "adam"
    lossfn = 'huber'
    layers = ['normalization', 'dropout', 200,'dropout', 150, 'dropout', 100, 'dropout',100, 'dropout',100, 'dropout',50,'dropout', 20]
    regularization_strength = 0
    activationfn = LeakyReLU(alpha = .1)
    epochs=25000

    batch_size=1000
    dropout_rate = .2

    models.loc[model_id, 'model_id'] = model_id
    models.loc[model_id, 'dataname'] = dataname
    models.loc[model_id, 'optimizer'] = optimizer
    models.loc[model_id, 'lossfn'] = lossfn
    models.loc[model_id, 'layers'] = str(layers)
    models.loc[model_id, 'regularization_strength'] = regularization_strength
    models.loc[model_id, 'activationfn'] = activationfn
    models.loc[model_id, 'epochs'] = epochs
    models.loc[model_id, 'batch_size'] = batch_size
    models.loc[model_id, 'dropout_rate'] = dropout_rate

    if (optimizer == "adam"):
        opt = keras.optimizers.Adam()

    # import and organize data
    filename = "data/datasets/" + dataname
    dataset = loadtxt(filename, delimiter=',', skiprows=1)
    np.random.shuffle(dataset)
    # split into input (X) and output (y) variables

    X = dataset[:, 1:-1]
    #X = dataset[:, (-1, -2)]
    d = X.shape[1]
    n = X.shape[0]
    y = dataset[:,-1]

    split = int(8*n/10)
    Xtrain = X[:split, :]
    Xtest = X[split+ 1:, :]

    ytrain = y[:split]
    ytest = y[split + 1:]

    print(ytrain)

    # define the keras model
    model = Sequential()

    for layer in layers:
        if (layer == 'normalization'):
            model.add(Normalization())
        elif layer == 'dropout':
            model.add(Dropout(dropout_rate))
        elif regularization_strength > 0:
            model.add(Dense(layer, activation=activationfn, 
                            kernel_regularizer=regularizers.L1L2(regularization_strength, regularization_strength), 
                            bias_regularizer=regularizers.L1L2(regularization_strength, regularization_strength)))
        else:
            model.add(Dense(layer, activation=activationfn))
    model.add(Dense(1, activation='linear'))

    # compile the keras model
    model.compile(loss=lossfn, optimizer=opt, metrics=['mean_squared_error'])
    # fit the keras model on the dataset
    start = time()
    model.fit(Xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=1)
    timee = time()-start
    # evaluate the keras model
    train_accuracy = model.evaluate(Xtrain, ytrain)
    test_accuracy = model.evaluate(Xtest, ytest)

    print('Train Accuracy:')
    print(train_accuracy)
    print('Test Accuracy:')
    print(test_accuracy)

    # save model info
    output = model.predict(Xtrain)
    print("ytrain")
    print(ytrain)
    print(ytrain.shape)
    print("output")
    print(output)
    print(output.shape)
    plt.hist(output, bins=40)
    plt.title("Distribution of Predicted SLR from Training Data")
    plt.xlabel("SLR")
    plt.savefig('prediction_distributions/' + str(model_id) + '_train.png')
    plt.clf()

    plt.scatter(ytrain, output)
    plt.title("Measured and Predicted SLR by Model on Training Data")
    plt.xlabel("Measured SLR")
    plt.ylabel("Predicted SLR")
    plt.savefig('prediction_distributions/' + str(model_id) + '_train_scatter.png')
    plt.clf()

    output = model.predict(Xtest)
    print("ytest")
    print(ytest)
    print("output")
    print(output)
    plt.hist(output, bins=40)
    plt.title("Distribution of Predicted SLR from Test Data")
    plt.xlabel("SLR")
    plt.savefig('prediction_distributions/' + str(model_id) + '_test.png')
    plt.clf()

    plt.scatter(ytest, output)
    plt.title("Measured and Predicted SLR by Model on Test Data")
    plt.xlabel("Measured SLR")
    plt.ylabel("Predicted SLR")
    plt.savefig('prediction_distributions/' + str(model_id) + '_test_scatter.png')
    plt.clf()

    print(models)

    models.loc[model_id, 'train_accuracy'] = str(train_accuracy)
    models.loc[model_id, 'test_accuracy'] = str(test_accuracy)
    models.loc[model_id, 'time'] = str(timee)

    models.to_csv('models.csv', index=False)
    model.save('models/' + str(model_id))