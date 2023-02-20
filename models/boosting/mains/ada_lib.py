### IMPORTS ###
from numpy import mean
import numpy as np
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import sys
import os
from itertools import product
from xgboost import XGBClassifier

# Check folders so it works for different OS:s
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting/mains", "")
sys.path.insert(1, os.path.join(dirname, "general_classes"))

from DataPreparation import DataPreparation
from Performance import Performance
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

### FUNCTIONS ###

def normal_pred():
 
    # Set data
    path_data = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path_data, drop_cols =[], numpy_bool = True, gender = False, normalize = True)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Use data augmentation
    sm = SMOTE(k_neighbors = 5)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))

    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Fit the model and make predictions
    model = AdaBoostClassifier(n_estimators=50, learning_rate = 1).fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    # Print performance
    print("AdaBoostClassifier")
    print(classification_report(Y_test, y_pred))


def cross_val():
    # Set data 
    path_data = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path_data, drop_cols =[], numpy_bool = True, gender = False, normalize = True)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Use data augmentation
    sm = SMOTE(k_neighbors = 5)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((Y_train, Y_test))
    
    # Normalize the data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Fit the model and use k-fold cross validation
    model = AdaBoostClassifier()

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_train, Y_train, scoring="accuracy", cv=cv, n_jobs=-1, error_score='raise')
     
    for i in range(len(n_scores)):
        print(f"Iteration {i}, accuracy:\t {n_scores[i]}")

    print(f"Mean accuracy:\t {np.mean(n_scores)}")
    """
    n_cols = X_train.shape[1]
    best_subset, best_score = None, 0.0
    # enumerate all combinations of input features
    for subset in product([True, False], repeat=n_cols):
        # convert into column indexes
        ix = [i for i, x in enumerate(subset) if x]
        # check for now column (all False)
        if len(ix) == 0:
            continue
            # select columns
        X_new = X_train[:, ix]
        # define model
        model = AdaBoostClassifier()
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate model
        scores = cross_val_score(model, X_new, Y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        # summarize scores
        result = mean(scores)
        # report progress
        print('>f(%s) = %f ' % (ix, result))
        # check if it is better than the best so far
        if best_score is None or result >= best_score:
            # better result
            best_subset, best_score = ix, result
            # report best
    print('Done!')
    print('f(%s) = %f' % (best_subset, best_score))
 
    """

def main():
    normal_pred()
    cross_val()

if __name__ == "__main__":
    main()



