
# evaluate adaboost algorithm for classification
from numpy import mean
import numpy as np
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
import sys
import os
from itertools import product

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting", "")
sys.path.insert(1, os.path.join(dirname, "general_classes"))
from DataPreparation import DataPreparation
from Performance import Performance





def main():
 
    # Define dataset
    path_data = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path_data, drop_cols = ["Mean Age Female", "Mean Age Male"], numpy_bool = True, gender = False)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((Y_train, Y_test))

    # Use SMOTE
    
    X_res, Y_res = DataPrep.SMOTE(num = None, perc = 300, k = 5, SMOTE_feature = -1)
    X_res2, Y_res2 = DataPrep.SMOTE(num = None, perc = 300, k = 5, SMOTE_feature = 1)

    X_train = np.concatenate((X_train, X_res))
    X_train = np.concatenate((X_train, X_res2))
    Y_train = np.concatenate((Y_train, Y_res))
    Y_train = np.concatenate((Y_train, Y_res2))
    """ 
    # Define the model
    model = AdaBoostClassifier()

    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X_train, Y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    
    # report performance
    print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))  
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

if __name__ == "__main__":
    main()



