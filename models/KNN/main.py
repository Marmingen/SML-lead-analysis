### IMPORTS ###
import sys
import os
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from general_classes import *
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_me

### TERMINAL ###
# os.system('cls')

# dirname = os.getcwd()
# dirname = dirname.replace("/models/boosting", "")
# sys.path.insert(1, os.path.join(dirname, "general_classes"))
# from DataPreparation import DataPreparation
# from Performance import Performance

### FUNCTIONS ###

def normalize(X):
        return (X - X.min())/(X.max() - X.min())

### MAIN ###

def main():
    
    # Import data from train.csv, in order (Y_train, X_train, X_val, Y_val)

    dp = DataPreparation('./data/train.csv', numpy_bool=False, gender=False)

    # Create input sets of X, output 'Lead' as Y and normalize

    X, Y = dp.raw()
    X = X.drop(columns=['Year', 'Mean Age Male', 'Mean Age Female'])

    Y = Y.replace("Female", -1)
    Y = Y.replace("Male", 1)

    # Check for zeros in the data sets

    # print(X.isnull().sum())     # No zero terms, OK
    # print(Y.isnull().sum())     # No zero terms, OK

    # Normalize input values

    normalize_X = normalize(X)

    # Split the data sets into 70% training data and 30% validation data

    X_train, X_val, Y_train, Y_val = skl_ms.train_test_split(normalize_X, Y, test_size = 0.3)

    # Implement k-NN algorithm for different k - values

    k = 75
    K = np.arange(1, k) 

    misclassification = []

    for k in K: 
        model = skl_nb.KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
        Y_pred = model.predict(X_val)
        misclassification.append(np.mean(Y_pred != Y_val))

    # Plot the results

    plt.figure(1)
    plt.plot(K, misclassification)
    plt.title('Training error using k-NN for different k')
    plt.xlabel('k')
    plt.ylabel('Error')


    ### Calculate the results for different validation sets and take the average ###

    n = 1

    misclassification_average = np.zeros((n, len(K)))
    
    # Create new training and validation set for each loop to take the average of it

    for i in range(n):

        X_train, X_val, Y_train, Y_val = skl_ms.train_test_split(normalize_X, Y, test_size = 0.3)

        for j, k in enumerate(K):
            model = skl_nb.KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
            Y_pred = model.predict(X_val)
            misclassification_average[i, j] = np.mean(Y_pred != Y_val)

    misclassification_average = np.mean(misclassification_average, axis=0)

    # Plot the results 

    plt.figure(2)
    plt.plot(K, misclassification_average)
    plt.title(f'Average ({n} diff. validation sets) error using k-NN for different k')
    plt.xlabel('k')
    plt.ylabel('Error')


    ### TUNING ### 

    # With n-fold cross validation 

    n_folds = 10 #6

    cross_val = skl_ms.KFold(n_splits = n_folds, shuffle= True)

    misclass_cross_val = np.zeros(len(K))

    # Create a loop where the cross validation changes order and calculate the error

    for index_train, index_val in cross_val.split(X):                       # cross_val.split() gives the indices for the training and validation data
        X_train, X_val = normalize_X.iloc[index_train], normalize_X.iloc[index_val]
        Y_train, Y_val = Y.iloc[index_train], Y.iloc[index_val]

        for j, k in enumerate(K):
            model = skl_nb.KNeighborsClassifier(n_neighbors = k).fit(X_train, Y_train)
            Y_pred = model.predict(X_val)
            misclass_cross_val[j] += np.mean(Y_pred != Y_val)

    misclass_cross_val /= n_folds

    plt.figure(3)
    plt.plot(K, misclass_cross_val)
    plt.title(f'Error for k-NN with, n = {n_folds} folds cross validation with different k-values')
    plt.ylabel('Error')
    plt.xlabel('k')

    ### EVALUATE THE RESULTS ###
    """ We calculated an approximative value for k = [10, 15]. So we set it to 10 and evaluate the results """
    
    X_train, X_val, Y_train, Y_val = skl_ms.train_test_split(normalize_X, Y, test_size = 0.3)

    fitted_model = skl_nb.KNeighborsClassifier(n_neighbors=10).fit(X_train, Y_train)
    Y_pred = fitted_model.predict(X_val)

    # Precision, recall, f1-score performance of the fitted model.
    print(skl_me.classification_report(Y_val, Y_pred))

    # Using ROC score to check our performance of the fitted model. 
    print(skl_me.roc_auc_score(Y_val, Y_pred))
    

    ### TUNING ###

    # Hyperparameters that could be useful to tune
    leaf_size = list(range(1, 20))
    n_neigh = list(range(5, 20))
    metric = ['minkowski', 'euclidean', 'manhattan']
    weights = ['uniform', 'distance']

    hyperpara = dict(leaf_size = leaf_size, n_neighbors=n_neigh, metric = metric, weights = weights)

    gs = skl_ms.GridSearchCV(skl_nb.KNeighborsClassifier(), hyperpara, cv=10, verbose=1, n_jobs=-1)
    
    best_model = gs.fit(X_train, Y_train)       # -> {'leaf_size': 1, 'metric': 'minkowski', 'n_neighbors': 10, 'weights': 'uniform'}

    print(best_model.best_params_)
    print(best_model.best_score_)

    # Test the new model with tuning with cross-validation
    # n_folds = 10

    # cross_val = skl_ms.KFold(n_splits = n_folds, shuffle= True)

    # misclass_tuned =np.zeros(n_folds)

    # # Create a loop where the cross validation changes order and calculate the error

    # for i, (index_train, index_val) in enumerate(cross_val.split(X)):                       # cross_val.split() gives the indices for the training and validation data
    #     X_train, X_val = normalize_X.iloc[index_train], normalize_X.iloc[index_val]
    #     Y_train, Y_val = Y.iloc[index_train], Y.iloc[index_val]

    #     tuned_model = skl_nb.KNeighborsClassifier(leaf_size=1, n_neighbors=10, weights='uniform', metric='minkowski').fit(X_train, Y_train)
    #     Y_pred_tuned = tuned_model.predict(X_val)
    #     misclass_tuned[i] += np.mean(Y_pred_tuned != Y_val)

    # print(np.mean(misclass_tuned))
    # plt.figure(4)
    # plt.boxplot(misclass_tuned)
    # plt.title('Tuned model with cross-validation')
    plt.show()

    # Cross-validation score
    tuned_model = skl_nb.KNeighborsClassifier(leaf_size=1, n_neighbors=10, weights='uniform', metric='minkowski').fit(X_train, Y_train)
    Y_pred_tuned = tuned_model.predict(X_val)
    cross_val_score = skl_ms.cross_val_score(tuned_model, normalize_X, Y, cv=10)

    print(f'Model accuracy: ', np.mean(cross_val_score))
    # Precision, recall, f1-score performance of the fitted model.
    print(skl_me.classification_report(Y_val, Y_pred_tuned))

    # Using ROC score to check our performance of the fitted model. 
    print(skl_me.roc_auc_score(Y_val, Y_pred_tuned))

### RUN (when in folder SML-LEAD-ANALYSIS) ###

if __name__ == '__main__':
    main()
