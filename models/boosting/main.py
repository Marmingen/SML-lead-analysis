### IMPORTS ###
import numpy as np
import sys
import os

from DecisionStump import DecisionStump
from AdaBoost import AdaBoost
from DataPreperation import DataPreperation

### CHECKING FOLDERS ###
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting", "")

sys.path.insert(1,os.path.join(dirname, "general_classes"))

### GLOBALS ###
clear = lambda : os.system("cls")

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
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
    

    clf = AdaBoost(n_clf=5)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print(f"Accuracy: \t{acc}")



if __name__ == "__main__":
    main()


