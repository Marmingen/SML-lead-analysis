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
from DataPreparation import DataPreparation
from Performance import Performance


### GLOBALS ###
clear = lambda : os.system("cls")

### MAIN ###

def main():
    # Fix data
    path_data = dirname + "/data/train.csv"
    drop_cols = ["Year", "Number words male", "Total words"]
    DataPrep = DataPreparation(path_data, numpy_bool = True, drop_cols = drop_cols, gender=False)
    # Add the to.numpy() converter in the adaboost. Check if it already is numpy or not
    X_train = DataPrep.X_train
    Y_train = DataPrep.Y_train
    X_test = DataPrep.X_test
    Y_test = DataPrep.Y_test

    # AdaBoost ML algortihm using 5 weak classifiers

    clf = AdaBoost(n_clf=5)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)


    # Analyze performance
    Perfor = Performance(y_pred, Y_test)
    accuracy = Perfor.accuracy()
    precision = Perfor.precision()
    recall = Perfor.recall()
    confusion = Perfor.confusion()
    f1 = Perfor.f1()
    cohen = Perfor.cohen()
    roc = Perfor.roc()

    print("Performance metrix\t\t")
    
    print(f"Accuracy: \t{accuracy}")
    print(f"Precision: \t{precision}")
    print(f"Recall: \t{recall}")
    print(f"Confusion: \t{confusion}")
    print(f"f1: \t{f1}")
    print(f"Cohen: \t{cohen}")
    print(f"Roc: \t{roc}")
    # Error due to the fact that AdaBoost class assumes inputs are np-arrays. Mine are pandas dataframes

if __name__ == "__main__":
    main()


