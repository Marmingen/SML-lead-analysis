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
    X_train = DataPrep.X_train
    Y_train = DataPrep.Y_train
    X_test = DataPrep.X_test
    Y_test = DataPrep.Y_test
    print("Check 1")

    # AdaBoost ML algortihm using 5 weak classifiers

    clf = AdaBoost(n_clf=5)
    print("Check 2")
    clf.fit(X_train, Y_train)
    print("Check 3")
    y_pred = clf.predict(X_test)
    print("Check 4")
    acc = accuracy(Y_test, y_pred)
    print("Check 5")
    print(f"Accuracy: \t{acc}")
    print("Check 6")

if __name__ == "__main__":
    main()


