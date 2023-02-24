### IMPORTS ###
import os
import sys
import numpy as np

# Check folders so it works for different OS:s
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting/mains", "")
sys.path.insert(1, os.path.join(dirname, "general_classes"))

from DataPreparation import DataPreparation
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from Performance import Performance
from sklearn.model_selection import KFold
import sklearn.metrics as skl_me

#gridsearch

def normal_pred():
    # Get the data sets
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, numpy_bool = True)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Use SMOTE for over sampling
    sm = SMOTE(random_state = 42)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    X_train = np.concatenate((X_train, X_res))
    Y_train = np.concatenate((Y_train, Y_res))

    # ML model
    GDB = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, \
            max_depth = 1, random_state = 0)
    model = GDB.fit(X_train, Y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    target_names = ['Female', 'Male']
    
    # Print statistics
    report = classification_report(Y_test, y_pred, target_names = target_names)
    print(report)
    perf = Performance(y_pred, Y_test)
    print(f"Cohen kappa: {perf.cohen()}")


def evaluation_cross_val(n_folds = 10):
    # Get the data sets
    path = dirname + "/data/train.csv"
    DataPrep = DataPreparation(path, numpy_bool = True)
    X_train, X_test, Y_train, Y_test = DataPrep.get_sets()

    # Merge the data
    X = np.concatenate((X_train, X_test))
    Y = np.concatenate((Y_train, Y_test))

    # Performance metrics
    accuracy = np.zeros(n_folds)
    balanced_accuracy = np.zeros(n_folds)
    precision = np.zeros(n_folds)
    recall_F = np.zeros(n_folds)
    recall_M = np.zeros(n_folds)
    F1 = np.zeros(n_folds)
    cohen_kappa = np.zeros(n_folds)
    
    GDB = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, \
            max_depth = 1, random_state = 0)

    
    cross_val = KFold(n_splits = n_folds, shuffle= True, random_state=False)

    for i, (index_train, index_val) in enumerate(cross_val.split(X)):     # cross_val.split() gives the indices for the training and validation data
        X_train_loop, X_val_loop = X[index_train], X[index_val]
        Y_train_loop, Y_val_loop = Y[index_train], Y[index_val]

        # Use SMOTE for over sampling
        sm = SMOTE(random_state = 42)
        X_res, Y_res = sm.fit_resample(X_train_loop, Y_train_loop)
        X_train_loop = np.concatenate((X_train_loop, X_res))
        Y_train_loop = np.concatenate((Y_train_loop, Y_res))
        
        model = GDB.fit(X_train_loop, Y_train_loop)

        Y_pred_loop = model.predict(X_val_loop)

        accuracy[i] = skl_me.accuracy_score(Y_val_loop, Y_pred_loop)
        balanced_accuracy[i] = skl_me.balanced_accuracy_score(Y_val_loop, Y_pred_loop)
        precision[i] = skl_me.precision_score(Y_val_loop, Y_pred_loop)
        recall_F[i] = skl_me.recall_score(Y_val_loop, Y_pred_loop, pos_label=-1)
        recall_M[i] = skl_me.recall_score(Y_val_loop, Y_pred_loop, pos_label=1)
        F1[i] = skl_me.f1_score(Y_val_loop, Y_pred_loop)
        cohen_kappa[i] = skl_me.cohen_kappa_score(Y_val_loop, Y_pred_loop)

    print(f"Mean accuracy: {np.mean(accuracy)}")
    print(f"Mean balanced accuracy: {np.mean(balanced_accuracy)}")
    print(f"Mean precision: {np.mean(precision)}")
    print(f"Mean recall F: {np.mean(recall_F)}")
    print(f"Mean recall M: {np.mean(recall_M)}")
    print(f"Mean f1: {np.mean(F1)}")
    print(f"Mean cohen kappa: {np.mean(cohen_kappa)}")



### MAIN ###

def main():
    normal_pred()
    evaluation_cross_val()
if __name__ == "__main__":
    main()
