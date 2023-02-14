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
    def __init__(self, path_data, numpy_bool = False, drop_cols = [], gender=False, random = False, normalize = False, clean=True):
        """
        path_data: absolute path to data
        numpy_bool: convert to numpy.ndarray or keep as pandas
        drop_cols: list of columns that should be dropped from dataframe
        gender: keep gender labels as strings or not
        random:
        normalize:
        clean: clean colinear data or not

        """
        self.numpy_bool = numpy_bool
        self.drop_cols = drop_cols
        self.gender = gender
        self.random = random
        self.normalize = normalize

        try:
            if sys.platform == "darwin": # for macOS
                self.data = pd.read_csv(os.path.join(dirname, path_data)) 
            else:
                self.data = pd.read_csv(path_data)
        except OSError as e:
            print("FileNotFoundError: [Errno 2] No such file or directory")
        
        
        if len(self.drop_cols) > 0:
            for col in drop_cols:
                self.data = self.data.drop([col], axis=1)

        if clean:
            self.__limit_vars()

        self.x_length = self.data.shape[0]
        self.y_length = self.data.shape[1]
        self.Y_train, self.X_train, self.X_test, self.Y_test = self.__create_data_sets()
        

    def get_sets(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def raw(self):
        X = self.data.drop(columns=['Lead'])
        Y = self.data['Lead']
        return X, Y
    
    def change_cols(self):
        return []

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
        
        if self.normalize:
            X_train = (X_train-X_train.min())/(X_train.max()-X_train.min())
            X_test = (X_test - X_test.min())/(X_test.max() - X_test.min())
        # add visualization methods

        # CLEAR COLUMNS AND PREPARE DATA

        if self.numpy_bool:
            return Y_train.to_numpy(), X_train.to_numpy(), \
                    X_test.to_numpy(), Y_test.to_numpy()
        else:
            return Y_train, X_train, X_test, Y_test

    def modify_cols(self):
        pass


    def visualize(self):
        pass
    
    def __limit_vars(self):
        
        # this selection was done since the VIF of the two were quite large
        # (logically so), thus theyre combined so that as little information is lost
        self.data["Lead age diff"] = self.data["Age Lead"] - self.data["Age Co-Lead"]
        
        # this selection was dine for the same reasons as above
        self.data["Mean age diff"] = self.data["Mean Age Male"] - self.data["Mean Age Female"]
        
        # logically, the amount of words features were going to be colinear, as seen by the
        # VIF-factors, thus theyre combined into three different features
        # (the fractions for lead and male have VIFs of ~<7, which is quite bad but
        # since some sources say VIFs<10 are acceptable and since we dont want to discard too
        # much data, theyre accepted as is)
        self.data["Fraction words female"] = self.data["Number words female"]/self.data["Total words"]
        self.data["Fraction words male"] = self.data["Number words male"]/self.data["Total words"]
        self.data["Fraction words lead"] = self.data["Number of words lead"]/self.data["Total words"]

        # if this turns out to increase k-fold accuracy, it stays
        self.data["Actor amount diff"] = self.data["Number of male actors"] - self.data["Number of female actors"]
        
        # the feature Year is omitted entirely partly due to it being multicolinear with 
        # features and partyl since it seems to have no large impact on the classification
        self.data = self.data.drop([""],axis=1)


    def k_fold(self, X_train, y_train, X_test, y_test, n_folds):
        if not isinstance(self.X_train, np.ndarray):
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy()
            X_test = X_test.to_numpy()
            y_test = y_test.to_numpy()

        X = np.concatenate((X_train, X_test))
        Y = np.concatenate((y_train, y_test))

        index = int(len(X)/n_folds)
        testing = []
        training = []

        for i in range(n_folds):
            X = np.concatenate((X_train, X_test))
            Y = np.concatenate((y_train, y_test))
            test_idx = [i for i in range(index*i, index*(i+1))]
            testing.append((X[test_idx], Y[test_idx]))
            training.append((np.delete(X, test_idx, axis=0), np.delete(Y, test_idx, axis=0)))


        return training, testing 
        

    def SMOTE(self, num = None, perc = None, k = 5, SMOTE_feature = -1):
        # num doesnt work
        """
        Synthetic minority over sampling technique
        
        Creates minority samples of a given class using k nearest neighbors
        Inputs:
            N: percentage of how many more samples should be generated
            k: k in k nearest neighbors
            SMOTE_feature: which class that synthetic samples are generated for

        """
        if num == None and perc == None:
            print("Need to specify absolute number of points, or percentage of points, that should be generated")
            return 0
        elif num != None and perc != None:
            print("Can't specify both absolute number of points, and percentage of points that should be generated")
            return 0

        # Create samples of the minority class
        min_sample_idx = np.where(self.Y_train == SMOTE_feature)[0]
        sample = pd.DataFrame(self.X_train[min_sample_idx])
        T, num_attrs = sample.shape

        if num != None:
            perc = int((num+T)/T)

            synthetic = np.zeros([perc*T, num_attrs])
        else:
            # If N is less than 100%, randomize the class samples
            if perc < 100:
                T = round(perc / 100 * T)
                perc = 100
            perc = int(perc / 100)
    
            synthetic = np.zeros([perc*T, num_attrs])
        new_index = 0
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(sample.values)
        
        # Populate the synthetic samples
        def populate(N, i, nnarray):
            nonlocal new_index
            nonlocal synthetic
            nonlocal sample
            while N != 0:
                nn = randrange(1, k+1)
                for attr in range(num_attrs):
                    dif = sample.iloc[nnarray[nn]][attr] - sample.iloc[i][attr]
                    gap = uniform(0, 1)
                    synthetic[new_index][attr] = sample.iloc[i][attr] + gap * dif
                new_index += 1              
                N = N - 1
        
        
        for i in range(T):
            nnarray = nbrs.kneighbors(sample.iloc[i].values.reshape(1, -1), return_distance=False)[0]
            populate(perc, i, nnarray)
        #print(synthetic)
        new_y = [SMOTE_feature for i in range(len(synthetic))]
        return synthetic, new_y
        
