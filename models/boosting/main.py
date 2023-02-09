### IMPORTS ###
import numpy as np
import sys
import os
sys.path.append(str(sys.path[0][:-14]))
from AdaBoost import AdaBoost


### CHECKING FOLDERS ###
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting", "")
sys.path.insert(1,os.path.join(dirname, "general_classes"))
from DataPreperation import DataPreperation


### GLOBALS ###
clear = lambda : os.system("cls")

def accuracy(Y_true, Y_pred):
    accuracy = np.sum(Y_true == Y_pred) / len(Y_true)
    return accuracy


### MAIN ###

def main():
    # Fix data
    path_data = dirname + "/data/train.csv"

    DataPrep = DataPreperation(path_data)
    # Add the to.numpy() converter in the adaboost. Check if it already is numpy or not
    X_train = DataPrep.X_train.to_numpy()
    Y_train = DataPrep.Y_train.to_numpy()
    X_test = DataPrep.X_test.to_numpy()
    Y_test = DataPrep.Y_test.to_numpy()
    
    # AdaBoost ML algortihm using 5 weak classifiers

    clf = AdaBoost(n_clf=5)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(Y_test, y_pred)
    
    print(f"Accuracy: \t{acc}")
    # Error due to the fact that AdaBoost class assumes inputs are np-arrays. Mine are pandas dataframes
if __name__ == "__main__":
    main()


