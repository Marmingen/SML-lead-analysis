### IMPORTS ###
import numpy as np
import sys
import os
import pandas as pd
sys.path.append(str(sys.path[0][:-14]))
from AdaBoost import AdaBoost
from matplotlib import pyplot as plt
from GradientBoosting import GradientBoosting
from statistics import mean
Grad = GradientBoosting()

### CHECKING FOLDERS ###
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting", "")
sys.path.insert(1,os.path.join(dirname, "general_classes"))
from DataPreparation import DataPreparation
from Performance import Performance
from sklearn.model_selection import StratifiedKFold


### GLOBALS ###
clear = lambda : os.system("cls")

### MAIN ###



from imblearn.over_sampling import SMOTE

def main():

    # Fix data
    path_data = dirname + "/data/train.csv"
    drop_cols = []
    DataPrep = DataPreparation(path_data, numpy_bool = True, drop_cols = drop_cols, gender=False)
    
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Create synthetic data
    X_res, Y_res = DataPrep.SMOTE(num = None, perc=200, k=5, SMOTE_feature = -1)
    X_res2, Y_res2 = DataPrep.SMOTE(num = None, perc=200, k=5, SMOTE_feature = 1)
    
    
    # Merge all the data
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))
    X_train = np.concatenate((X_train, X_res2))
    Y_train = np.concatenate((Y_train, Y_res2))
    
    #sm = SMOTE(random_state = 42)
    #X_res, Y_res = sm.fit_resample(X_train, Y_train)
    #X_train = np.concatenate((X_train, X_res))
    #Y_train = np.concatenate((Y_train, Y_res))
    r_indeces = np.arange(len(X_train))
    np.random.shuffle(r_indeces)
    X_train = X_train[r_indeces]
    Y_train = Y_train[r_indeces]
     
    training, testing = DataPrep.k_fold(X_train, Y_train, X_test, Y_test,5)
    scores = []
    for i in range(len(training)):
        print("****************************")
        print(f"Iteration number: \t {i}")
        clf = AdaBoost(n_clf = 5)
        clf.fit(training[i][0], training[i][1])
        y_pred = clf.predict(testing[i][0])
        Perfor = Performance(y_pred, testing[i][1])
        acc = Perfor.accuracy()
        scores.append(acc)
        print(f"Accuracy: \t {acc}")
        print("****************************")
        print(f"Females:\t {len(np.where(training[i][1]==-1)[0])}")
        print(f"Males:\t {len(np.where(training[i][1] == 1)[0])}")

    print(f"Average accuracy: \t{mean(scores)}")
    """
    sk_Kfold = StratifiedKFold(n_splits = 10)
    kFold = sk_Kfold.split(X_train, Y_train)

    scores = []
    for k, (train, test) in enumerate(kFold):
        print(f"Iteration\t {k}")
        clf = AdaBoost(n_clf = 5)
        clf.fit(X_train[train,:], Y_train[test])
        y_pred = clf.predict(X_test)
        Perfor = Performance(y_pred, Y_test)
        acc = Perfor.accuracy()
        scores[k] = acc
        print(f"Iteration {k}, accuracy: \t {acc}")
    print(scores)
    """
    
    """ 
    DataPrep.k_fold(5)
    
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
    #print(f"Precision: \t{precision}")
    #print(f"Recall: \t{recall}")
    #print(f"Confusion: \t{confusion}")
    #print(f"f1: \t{f1}")
    #print(f"Cohen: \t{cohen}")
    #print(f"Roc: \t{roc}")
    """

if __name__ == "__main__":
    main()


