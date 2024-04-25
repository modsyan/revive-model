"""
The file loads the data for full-body skeletons for the Deep Squat exercise

There are 90 correct sequences: data_correct has 90 training sequences (each consisting of 240 frames x 117 axes)

There are 90 incorrect sequences: data_incorrect has 90 training sequences (each consisting of 240 frames x 117 axes)

The movement sequences are loaded from the Data folder, saved in csv format

"""

import csv
import numpy as np

def load_data(self, correct_seq_num, incorrect_seq_num, n_dim_correct, n_dim_incorrect, n_frames):
    f = open('../../../../data_output/Data_Correct.csv')
    csv_f = csv.reader(f)
    X_Corr = list(csv_f)

    # Convert the input sequences into numpy arrays
    train_input1 = np.asarray(X_Corr)

    data_correct = np.zeros((correct_seq_num,n_frames,n_dim_correct))
    for i in range(len(train_input1)//n_dim_correct):
          data_correct[i,:,:] = np.transpose(train_input1[n_dim_correct*i:n_dim_correct*(i+1),:])
    
    f = open('../../../../data_output/Data_Incorrect.csv')
    csv_f = csv.reader(f)
    X_Incor = list(csv_f)

    # Convert the input sequences into numpy arrays
    train_input2 = np.asarray(X_Incor)
    data_incorrect = np.zeros((incorrect_seq_num,n_frames,n_dim_incorrect))
    for i in range(len(train_input2)//n_dim_incorrect):
          data_incorrect[i,:,:] = np.transpose(train_input2[n_dim_incorrect*i:n_dim_incorrect*(i+1),:])
    
    return data_correct, data_incorrect