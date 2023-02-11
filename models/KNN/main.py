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
import general_classes.Performance as performance

### TERMINAL ###
os.system('cls')

# dirname = os.getcwd()
# dirname = dirname.replace("/models/boosting", "")
# sys.path.insert(1, os.path.join(dirname, "general_classes"))
# from DataPreparation import DataPreparation
# from Performance import Performance

### FUNCTIONS ###


### MAIN ###

def main():
    
    # Import data from train.csv, in order (Y_train, X_train, X_test, Y_test)

    dp = DataPreparation('./data/train.csv', numpy_bool=False, gender=False)

    # Create (if numpy_bool=False) pandas frames for 70% training data and 30% validation data

    Y_train, X_train, X_test, Y_test = dp.create_data_sets()
    
    # Implement k-NN algorithm for different k - values

    k = 200
    K = np.arange(1, k) 

    misclassification = []

    for k in K: 
        model = skl_nb.KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        misclassification.append(np.mean(Y_pred != Y_test))

    # Plot the results

    plt.figure(1)
    plt.plot(K, misclassification)
    plt.title('Training error using k-NN for different k')
    plt.xlabel('k')
    plt.ylabel('Error')


    ### 
    ### Calculate the results for different validation sets and take the average ###

    n = 10

    misclassification_average = np.zeros((n, len(K)))
    
    # Import the dataset again but for random_seed off so the validation sets become different

    df = DataPreparation('./data/train.csv', numpy_bool=True, gender=False, random=True)

    for i in range(n):

        Y_train, X_train, X_test, Y_test = df.create_data_sets()

        for j, k in enumerate(K):
            model = skl_nb.KNeighborsClassifier(n_neighbors=k).fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            misclassification_average[i, j] = np.mean(Y_pred != Y_test)

    misclassification_average = np.mean(misclassification_average, axis=0)

    # Plot the results 

    plt.figure(2)
    plt.plot(K, misclassification_average)
    plt.title('Average (diff. validation sets) error using k-NN for different k')
    plt.xlabel('k')
    plt.ylabel('Error')
    plt.show()


    ### TUNING ### 

    # With n-fold cross validation 

    n_folds = 10

    cross_val = skl_ms.KFold(n_splits = n_folds, )

    misclass_cross_val = np.zeros(len(K))

    for train_i, test_i in cross_val.split(X_train):
        


        

### RUN (when in folder SML-LEAD-ANALYSIS) ###

if __name__ == '__main__':
    main()
