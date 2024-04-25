import project.constants as cn
import project.utils.file_system as fs
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def ensure_exercise_exists(ex_dir_name):
    fs.create_dir_if_not_exists(cn.UI_PRMD_CORRECT_SEGMENTED_MOVEMENTS)
    fs.ensure_dir_exists(ex_dir_name)


def load_data(subject, episode, correct=True):
    prefix = "Segmented Movements/Vicon/Angles/" if correct else "Incorrect Segmented Movements/Vicon/Angles/"
    filename = f"m01_s{subject:02d}_e{episode:02d}_angles.txt"
    path = prefix + filename
    return np.loadtxt(path, delimiter=',')


n_subjects = 10  # num_of_exercises
n_episodes = 10  # numb_of_reputation_per_exersice


class UIPRMDDataPreparation:

    def __init__(self):
        self.subjects_no = n_subjects
        self.episodes_no = n_episodes

    # def import_dataset(self):
    #     train_x = pd.read_csv(self.dir + "/Train_X.csv", header=None).iloc[:, :].values
    #     train_y = pd.read_csv(self.dir + "/Train_Y.csv", header=None).iloc[:, :].values
    #     return train_x, train_y

    def correct_data_preparation(self, ex_path, output_path=""):
        ensure_exercise_exists(ex_path)

        # Load data for correct sequences
        correct_data = [load_data(s, e) for s in range(1, n_subjects + 1)
                        for e in range(1, n_episodes + 1)]

        # Perform linear alignment
        incorrect_data = [load_data(s, e, correct=False) for s in range(1, n_subjects + 1)
                          for e in range(1, n_episodes + 1)]

        # Perform linear alignment
        length_mean = 240
        correct_aligned = [interp1d(np.arange(len(seq)), seq, axis=0)(np.linspace(0, len(seq) - 1, length_mean)) for seq
                           in correct_data]
        incorrect_aligned = [interp1d(np.arange(len(seq)), seq, axis=0)(np.linspace(0, len(seq) - 1, length_mean)) for
                             seq in incorrect_data]

        # Transform into matrices
        correct_xm = np.concatenate(correct_aligned)
        incorrect_xm = np.concatenate(incorrect_aligned)

        # Re-center the data to obtain zero mean
        data_mean = np.mean(correct_xm, axis=1)
        centered_correct_data = correct_xm - data_mean[:, np.newaxis]
        data_mean_inc = np.mean(incorrect_xm, axis=1)
        centered_incorrect_data = incorrect_xm - data_mean_inc[:, np.newaxis]

        # Scale the data between -1 and 1
        scaling_value = np.ceil(np.max([np.max(centered_correct_data), np.max(np.abs(centered_correct_data))]))
        data_correct = centered_correct_data / scaling_value
        data_incorrect = centered_incorrect_data / scaling_value

        # Save the data
        np.savetxt(f'{output_path}/Data_Correct.csv', data_correct, delimiter=',')
        np.savetxt(f'{output_path}/Data_Incorrect.csv', data_incorrect, delimiter=',')

        return data_correct, data_incorrect

      def incorrect_data_preparation(self, autoencoder_path):
        ensure_exercise_exists(ex_path)

        """ 
            %% Load Autoencoder files
        """

        data_nn = np.loadtxt('Autoencoder_Output_Correct.csv', delimiter=',')

        Data_NN_inc = np.loadtxt('Autoencoder_Output_Incorrect.csv', delimiter=',')

        """
            %% Reshape the data for GMM 
        """

        nDim = 4  # % The output dimension of the autoencoder is 4
        l = data_nn.shape[1] // nDim  # % number of frames in each movement repetition

        # Incorrect sequences

        n_seq_corr = data_nn.shape[0]  # % number of correct sequences

        data = np.tile(np.arange(1, l + 1), n_seq_corr)
        data_position = np.empty((nDim, 0))

        # % create a row for the time indices of correct data
        for i in range(n_seq_corr):
            temp = np.empty((0, nDim))
            for j in range(nDim):
                temp = np.vstack((temp, data_nn[i, j:nDim * l:nDim]))
            data_position = np.hstack((data_position, temp))

        data = np.vstack((data, data_position))

        # % Incorrect sequences

        n_seq_inc = Data_NN_inc.shape[0]
        Data_inc = np.tile(np.arange(1, l + 1), n_seq_inc)
        Data_inc_position = np.empty((nDim, 0))

        for i in range(n_seq_inc):
            temp = np.empty((0, nDim))
            for j in range(nDim):
                temp = np.vstack((temp, Data_NN_inc[i, j:nDim * l:nDim]))
            Data_inc_position = np.hstack((Data_inc_position, temp))

        Data_inc = np.vstack((Data_inc, Data_inc_position))

        """
            %% Train GMM
        """
        nbStates = 6
        nbVar = data.shape[0]
        Priors = np.zeros(nbStates)
        Mu = np.zeros((nbVar, nbStates))
        Sigma = np.zeros((nbVar, nbVar, nbStates))

        # % Define the number of states for GMM
        nb_states = 6
        #
        # % Number of rows in the data matrix
        # nbVar = size(Data,1);
        #
        # % Training by EM algorithm, initialized by k-means clustering
        # [Priors, Mu, Sigma] = EM_init_regularTiming(Data, nbStates);
        # [Priors, Mu, Sigma] = EM_boundingCov(Data, Priors, Mu, Sigma);

    def prepare_data(self):
        x = ""
