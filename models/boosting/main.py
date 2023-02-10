### IMPORTS ###
import numpy as np
import sys
import os
sys.path.append(str(sys.path[0][:-14]))
from AdaBoost import AdaBoost
from matplotlib import pyplot as plt


### CHECKING FOLDERS ###
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting", "")
sys.path.insert(1,os.path.join(dirname, "general_classes"))
from DataPreperation import DataPreperation
from Performance import Performance


### GLOBALS ###
clear = lambda : os.system("cls")

### MAIN ###

def main():
    # Fix data
    path_data = dirname + "/data/train.csv"

    DataPrep = DataPreperation(path_data, numpy_bool = True)
    # Add the to.numpy() converter in the adaboost. Check if it already is numpy or not
    X_train = DataPrep.X_train
    Y_train = DataPrep.Y_train
    X_test = DataPrep.X_test
    Y_test = DataPrep.Y_test

    # AdaBoost ML algortihm using 5 weak classifiers

    clf = AdaBoost(n_clf=5)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)

    Perfor = Performance(y_pred, Y_test)
    acc = Perfor.accuracy()
    
    print(f"Accuracy: \t{acc}")
    # Error due to the fact that AdaBoost class assumes inputs are np-arrays. Mine are pandas dataframes
if __name__ == "__main__":
    main()


