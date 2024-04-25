import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from keras.layers import concatenate, Flatten, Dropout, Dense, Input, LSTM
from keras.models import Model
from keras.optimizers import *

import matplotlib.pyplot as plt

from math import sqrt

from data_processing import Data_Loader, Test_Data_Loader


from graph import Graph

from sgcn_lstm import Stgcn_Lstm

from sklearn.metrics import mean_squared_error, mean_absolute_error

import PyKinectBodyGame


random_seed = 42  # for reproducibility

data_loader = Data_Loader("Kimore ex5")
graph = Graph(len(data_loader.body_part))
train_x, valid_x, train_y, valid_y = train_test_split(data_loader.scaled_x, data_loader.scaled_y, test_size=0.2, random_state = random_seed)
print("Training instances: ", len(train_x))
print("Validation instances: ", len(valid_x))
algorithm = Stgcn_Lstm(train_x, train_y, valid_x, valid_y, graph.AD, graph.AD2, epoach = 1000)


test_data_loader = Test_Data_Loader("Test_ex5")
model = algorithm.build_model()
model.load_weights("best model/best_model_ex5.hdf5")


predictions = []
for i in range(test_data_loader.scaled_x.shape[0]): 

    data  = test_data_loader.scaled_x[i]
    prediction = model.predict(data.reshape(1,data.shape[0],data.shape[1],data.shape[2]))

    predictions.append(prediction[0,0])

final_prediction = sum(predictions[1:]) / (len(predictions)-1)  

file1 = open('prediction.txt', 'w')
file1.write(str(final_prediction))