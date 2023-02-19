##########################################################
## IMPORTS

import numpy as np
import pandas as pd
import sys
import os
from random import uniform
from random import randrange
from sklearn.neighbors import NearestNeighbors
from scipy import stats

##########################################################
## CHECKING FOLDERS

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/general_classes", "")

##########################################################
## GLOBALS

clear = lambda : os.system("cls")

##########################################################
## DataPreparation CLASS


class DataPreparation():
    
    def __init__(self, path_data, numpy_bool = False, gender=False,
                 random = False, normalize = False, clean=True, custom = False):
        """
        :param str path_data: absolute path to data
        :param bool numpy_bool: convert to numpy.ndarray or keep as pandas
        :param bool gender: keep gender labels as strings or not
        :param bool random:
        :param bool normalize:
        :param bool clean: clean colinear data or not
        :param bool custom: use custom parameters for the data features or not
        
        """
        
        self.numpy_bool = numpy_bool
        self.gender = gender
        self.random = random
        self.normalize = normalize

        # for OS-check
        try:
            if sys.platform == "darwin": # for macOS
                self.data = pd.read_csv(os.path.join(dirname, path_data)) 
            else:
                self.data = pd.read_csv(path_data)
        except OSError as e:
            print("FileNotFoundError: [Errno 2] No such file or directory")
<<<<<<< HEAD
        


        self.data["perc female words"] = self.data["Number words female"]/self.data["Total words"]
        self.data["perc male words"] = self.data["Number words male"]/self.data["Total words"]
        self.data["diff perc"] = self.data["Difference in words lead and co-lead"]/self.data["Total words"]
        self.data["perc female actors"] = self.data["Number of female actors"]/self.data["Number of male actors"]
        
        self.data["diff fem age"] = self.data["Age Lead"] - self.data["Mean Age Female"]
        self.data["diff man age"] = self.data["Age Lead"] - self.data["Mean Age Male"]
        #self.data["diff age"] = self.data["Mean Age Female"]/self.data["Mean Age Male"]
        #self.data["mean age"] = self.data["Age Lead"]/(self.data["Mean Age Female"] + self.data["Mean Age Male"])/2
        


        del self.data["Age Lead"]
        del self.data["Mean Age Female"]
        del self.data["Mean Age Male"]
        del self.data["Total words"]
        del self.data["Difference in words lead and co-lead"]
        del self.data["Gross"]
        del self.data["Year"]
        
        #mean age och perc female actors är skumma

        #del self.data["Number of words lead"]
        #del self.data["Number words female"]
        #del self.data["Number words male"]
        #del self.data["Number of female actors"]
        #del self.data["Number of male actors"]
        #del self.data["Age Co-Lead"]

        print(self.data.shape)
        print(self.data.columns)



        if len(self.drop_cols) > 0:
            for col in drop_cols:
                self.data = self.data.drop([col], axis=1)
=======

        # preparing the data
        if clean:
            self.__limit_vars()
        elif custom:
            self.__customized_vars()
>>>>>>> ad804886db5030bdd1fcbe614ae728feade1289e

        self.x_length = self.data.shape[0]
        self.y_length = self.data.shape[1]
        self.Y_train, self.X_train, self.X_test, self.Y_test = self.__create_data_sets()
    
    ##########################################################
    ## SECRET METHODS
    
     # for custom variables
    def __customized_vars(self):
        pass
    
    def __limit_vars(self):
        # this selection was done since the VIF of the features were quite large
        # (logically so), thus theyre combined so that as little information is lost
        self.data["Diff age"] = self.data["Mean Age Female"]/self.data["Mean Age Male"]
        self.data["Mean age"] = self.data["Age Lead"]/(self.data["Mean Age Female"] + self.data["Mean Age Male"])*2

        
        # logically, the amount of words features were going to be colinear, as seen by the
        # VIF-factors, thus theyre combined into two different features
        # (the fractions for lead and male have VIFs of ~<7, which is quite bad but
        # since some sources say VIFs<10 are acceptable and since we dont want to discard too
        # much data, theyre accepted as is)
        self.data["Frac female words"] = self.data["Number words female"]/self.data["Total words"]
        self.data["Frac male words"] = self.data["Number words male"]/self.data["Total words"]
        
        # if this turns out to increase k-fold accuracy, it stays
        self.data["Diff frac"] = self.data["Difference in words lead and co-lead"]/self.data["Total words"]
        self.data["Frac female actors"] = self.data["Number of female actors"]/self.data["Number of male actors"]
        
        
        # the feature Year is omitted entirely partly due to it being multicolinear with 
        # features and partyl since it seems to have no large impact on the classification
                                    
        self.data = self.data.drop(["Age Lead", "Mean Age Male", "Mean Age Female", "Total words",
                                    "Difference in words lead and co-lead", "Year"],axis=1)

    def __create_data_sets(self):
        if self.random:
            train = self.data.sample(frac= .7)
        else:
            train = self.data.sample(frac = .7, random_state=10)
        test = self.data.drop(train.index)

        Y_train = train["Lead"]
        X_train = train.drop("Lead", axis=1)
        
        if not self.gender:
            Y_train = Y_train.replace("Female", -1)
            Y_train = Y_train.replace("Male", 1)

        Y_test = test["Lead"]
        X_test = test.drop("Lead", axis=1)
        
        if not self.gender:
            Y_test = Y_test.replace("Female", -1)
            Y_test = Y_test.replace("Male", 1)
        
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

    ##########################################################
    ## ACCESSABLE METHODS

    def get_sets(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def raw(self):
        X = self.data.drop(columns=['Lead'])
        Y = self.data['Lead']
        return X, Y
    
    def change_cols(self):
        return []

    def modify_cols(self):
        pass

    def visualize(self):
        pass

    # might be deprecated
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
        
        print(min_sample_idx)
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
        new_y = [SMOTE_feature for _ in range(len(synthetic))]
        return synthetic, new_y
        
