##########################################################
## IMPORTS

import numpy as np
from sklearn import preprocessing

##########################################################
## FIX PATH

import sys
sys.path.append('.')

##########################################################
## LOCAL PACKAGES
from general_classes import *

##########################################################
## MAIN

def main():
    # setting up dataprep instance
    dp = DataPreparation("./data/train.csv", clean=False)
        
    # combining all of the code, since there will be no validation
    X_train, X_test, Y_train, Y_test = dp.get_sets()
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((Y_train, Y_test))
    
    # normalizing the data
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    
    # naive classifier predicting all men
    y_predict = np.ones(Y_test.shape)
    
    # setting up performance instance
    perf = Performance(y_predict, Y_test)
    
    # printing the performance
    dic = get_dict()
    perf.combination(dic)
    print_combination(dic)
    
##########################################################
## RUN CODE

if __name__ == "__main__":
    main()