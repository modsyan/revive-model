"""
The file loads the data for full-body skeletons

There are 63 correct sequences: data_correct has 63 training sequences (each consisting of 229 frames x 117 axes)

There are 63 incorrect sequences: data_incorrect has 63 training sequences (each consisting of 229 frames x 117 axes)

The movement sequences are loaded from the data_output folder, saved in csv format

"""

import csv
import numpy as np


def load_data():
    f = open('../data/Data_Correct.csv')
    csv_f = csv.reader(f)
    X_Corr = list(csv_f)

    # Convert the input sequences into numpy arrays
    train_input1 = np.asarray(X_Corr)

    n_dim = 117
    n_frames = 229
    n_correct_sequences = 36

    data_correct = np.zeros((n_correct_sequences, n_frames, n_dim))
    for i in range(len(train_input1) // n_dim):
        data_correct[i, :, :] = np.transpose(train_input1[n_dim * i:n_dim * (i + 1), :])

    f = open('../data/Data_Incorrect.csv')
    csv_f = csv.reader(f)
    X_Incor = list(csv_f)

    # Convert the input sequences into numpy arrays
    train_input2 = np.asarray(X_Incor)
    n_dim = 117
    data_incorrect = np.zeros((63, 229, n_dim))
    for i in range(len(train_input2) // n_dim):
        data_incorrect[i, :, :] = np.transpose(train_input2[n_dim * i:n_dim * (i + 1), :])

    return data_correct, data_incorrect
