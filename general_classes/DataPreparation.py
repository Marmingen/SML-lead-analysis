### IMPORTS ###

import numpy as np
import pandas as pd
import sys
import os
from random import uniform
from random import randrange
from sklearn.neighbors import NearestNeighbors

### CHECKING FOLDERS ###

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/general_classes", "")

### GLOBALS ###

clear = lambda : os.system("cls")


class DataPreparation():
    def __init__(self, path_data, numpy_bool = False, drop_cols = [], gender=False, random = False):
        """
        path_data: absolute path to data
        numpy_bool: convert to numpy.ndarray or keep as pandas
        drop_cols: list of columns that should be dropped from dataframe

        """
        self.numpy_bool = numpy_bool
        self.drop_cols = drop_cols
        self.gender = gender
        self.random = random

        try:
            if sys.platform == "darwin": # for macOS
                self.data = pd.read_csv(os.path.join(dirname, path_data)) 
            else:
                self.data = pd.read_csv(path_data)
        except OSError as e:
            print("FileNotFoundError: [Errno 2] No such file or directory")

        self.x_length = self.data.shape[0]
        self.y_length = self.data.shape[1]
        self.Y_train, self.X_train, self.X_test, self.Y_test = self.__create_data_sets()
        

    def get_sets(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test


    def raw(self):
        X = self.data.drop(columns=['Lead'])
        Y = self.data['Lead']
        return X, Y


    def __create_data_sets(self):
        if self.random:
            train = self.data.sample(frac= .7)
        else:
            train = self.data.sample(frac = .7, random_state=10)
        test = self.data.drop(train.index)

        Y_train = train["Lead"]
        if not self.gender:
            Y_train = Y_train.replace("Female", -1)
            Y_train = Y_train.replace("Male", 1)
        X_train = train.drop("Lead", axis=1)


        Y_test = test["Lead"]
        if not self.gender:
            Y_test = Y_test.replace("Female", -1)
            Y_test = Y_test.replace("Male", 1)
        X_test = test.drop("Lead", axis=1)

        # add visualization methods

        # CLEAR COLUMNS AND PREPARE DATA

        if self.numpy_bool:
            return Y_train.to_numpy(), X_train.to_numpy(), \
                    X_test.to_numpy(), Y_test.to_numpy()
        else:
            return Y_train, X_train, X_test, Y_test


    def __clean_data(self):
        pass
        
    
    def __Populate(self, N, i, nnarray, k, num_attrs, minority_sample, synthetics, new_index):
        """


        """
        while N != 0:
            nn = randrange(1, k+1)

            for attr in range(num_attrs):
                dif = minority_sample.iloc[nnarray[nn]][attr] - minority_sample.iloc[i][attr]
                gap = uniform(0, 1)
                synthetics[new_index][attr] = minority_sample.iloc[i][attr] + gap * dif
            new_index = new_index + 1
            N = N - 1

        return synthetics
    

    def SMOTE(self, N, k):
        """
        Synthetic minority oversampling technique. 
        Used for oversampling the minority feature: females
        T: Number of minority class samples
        N: Amount of SMOTE
        k: Number of nearest neighbours
        """

        # convert to np.ndarray data type
        if isinstance(self.X_train, np.ndarray):
            sample_x = pd.DataFrame(self.X_train)
            sample_y = pd.DataFrame(self.Y_train)
        else:
            sample_x = self.X_train
            sample_y = self.Y_train


        sample = sample_x.assign(Lead = sample_y.to_numpy())
        minority_sample = sample[sample["Lead"] == -1].drop(["Lead"], axis = 1)
        # get samples of minority class from the training set
        #minority_idx = [i for i in range(0, sample_y.shape[0]) if sample_y.iloc[i,0] == -1]
        #minority_sample = [sample_x.iloc[i, :] for i in minority_idx]
        #print(minority_sample) 
         
        T, num_attrs = minority_sample.shape # num_attrs: number of features
        if N < 100:
            T = round((N/100) * T)
            N = 100

        N = int(N/100)
        
        new_index = 0
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(minority_sample.values)
        synthetics = np.zeros([T * N, num_attrs])
        
        for i in range(T):
            #compute k nearest neighbours for i, and save the indeces to nnarray
            nnarray = nbrs.kneighbors(minority_sample.iloc[i].values.reshape(1, -1), return_distance = False)[0]

            synthetics = self.__Populate(N, i, nnarray, k, num_attrs, minority_sample, synthetics, new_index)

        return synthetics

    
    def visualize(self):
        pass


    def k_fold_cross_val(self, n_folds, n_rep, ):
        pass
        


