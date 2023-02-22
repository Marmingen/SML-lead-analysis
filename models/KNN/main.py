##########################################################
## IMPORTS

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_me

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

##########################################################
## FIXING PATH

import sys
sys.path.append('.')

##########################################################
## FUNCTIONS

def pre_proc(data):
    """ Minimize the features to exclude noice, many of them were correlated. k-NN is also sensitive for many features. """

    data['Diff frac male and female'] = data["Number words male"]/data["Total words"] - data["Number words female"]/data["Total words"]
    data["Fraction words lead"] = data["Number of words lead"]/data["Total words"]
    data['Fraction diff words lead and co-lead'] = data['Difference in words lead and co-lead']/data['Total words']
    data["Mean age diff"] = data["Mean Age Male"] - data["Mean Age Female"]
    data['Diff age lead and co-lead'] = data['Age Lead'] - data['Age Co-Lead']
    data['Diff number actors'] = data['Number of male actors'] - data['Number of female actors']

    data = data.drop(columns=['Number words female', 'Number words male', 'Total words', 'Difference in words lead and co-lead',
                            'Mean Age Male', 'Mean Age Female', 'Age Lead', 'Age Co-Lead', 'Number of male actors', 'Number of female actors', 'Number of words lead'])

    data = data.drop(columns=['Year', 'Gross'])

    return data

def scale_X(data, scaler = 1):
    """ Normalize the data with different scalers.
    1 ~ StandardScaler
    2 ~ MinMaxScaler
    3 ~ RobustScaler
    """

    if scaler == 1:
        scaler = StandardScaler()
    elif scaler == 2:
        scaler = MinMaxScaler()
    elif scaler == 3:
        scaler = RobustScaler()

    data = scaler.fit_transform(data)

    return data

def gender_num(data):
    """ Replace the gender with -1 for female, respectively 1 for male"""
    data = data.replace('Female', -1)
    data = data.replace('Male', 1)
    return data

def sep_X_Y(data):
    """ Separate the data frame into inputs (features), X, and output, Y """

    Y = data['Lead']
    X = data.drop(columns=['Lead'])
    return X, Y

def train_smote(X_train, Y_train):
    """ To balance an unbalanced data set using SMOTE """

    X_train, Y_train = SMOTE().fit_resample(X_train, Y_train)

    return X_train, Y_train

def evaluation_cross_val(X, Y, n_folds = 10, tuned=True):
    """ Evaluates k-NN model with cross validation """

    accuracy = np.zeros(n_folds)
    balanced_accuracy = np.zeros(n_folds)
    precision_F = np.zeros(n_folds)
    precision_M = np.zeros(n_folds)
    recall_F = np.zeros(n_folds)
    recall_M = np.zeros(n_folds)
    F1 = np.zeros(n_folds)
    cohen_kappa = np.zeros(n_folds)

    
    cross_val = skl_ms.KFold(n_splits = n_folds, shuffle= True, random_state=False)

    for i, (index_train, index_val) in enumerate(cross_val.split(X)):     # cross_val.split() gives the indices for the training and validation data
        X_train_loop, X_val_loop = X[index_train], X[index_val]
        Y_train_loop, Y_val_loop = Y[index_train], Y[index_val]

        # Use smote to balance the inbalanced data
        
        X_train_loop, Y_train_loop = train_smote(X_train_loop, Y_train_loop)

        if tuned:
            model = skl_nb.KNeighborsClassifier(metric='manhattan', n_neighbors=6, weights='distance').fit(X_train_loop, Y_train_loop)
        else:
            model = skl_nb.KNeighborsClassifier().fit(X_train_loop, Y_train_loop)

        Y_pred_loop = model.predict(X_val_loop)

        # Store the results

        accuracy[i] = skl_me.accuracy_score(Y_val_loop, Y_pred_loop)
        balanced_accuracy[i] = skl_me.balanced_accuracy_score(Y_val_loop, Y_pred_loop)
        precision_F[i] = skl_me.precision_score(Y_val_loop, Y_pred_loop, pos_label=-1)
        precision_M[i] = skl_me.precision_score(Y_val_loop, Y_pred_loop, pos_label=1)
        recall_F[i] = skl_me.recall_score(Y_val_loop, Y_pred_loop, pos_label=-1)
        recall_M[i] = skl_me.recall_score(Y_val_loop, Y_pred_loop, pos_label=1)
        F1[i] = skl_me.f1_score(Y_val_loop, Y_pred_loop)
        cohen_kappa[i] = skl_me.cohen_kappa_score(Y_val_loop, Y_pred_loop)

    return np.mean(accuracy), np.mean(balanced_accuracy), np.mean(precision_F), np.mean(precision_M), np.mean(recall_F), np.mean(recall_M), np.mean(F1), np.mean(cohen_kappa)

##########################################################
## MAIN

def main():
    """Set random seed to easier tune the model"""

    np.random.seed(10)


    """Import data from train.csv """
    
    data = pd.read_csv('./data/train.csv')


    """ Pre-processing the data """

    data = pre_proc(data)


    """ Replace the gender string to int, Male = 1, Female = -1 """

    data = gender_num(data)


    """Split the data into inputs, X, and output 'Lead' as Y.""" 
    X, Y = sep_X_Y(data)


    """Scale features values, because k-NN is a distance-based method """

    X = scale_X(X, scaler = 2)


    """ Evaluate model without tuning, default k = 5. Model implemented in evaluation_cross_val function """

    unt_accuracy, unt_bal_accuracy, unt_precision_F, unt_precision_M, unt_recall_F, unt_recall_M, unt_f1, unt_cohen_kappa = evaluation_cross_val(X, Y, n_folds=10, tuned=False)

    print('Untuned model:')
    print(f'Accuracy score: {unt_accuracy}')
    print(f'Balanced accuracy: {unt_bal_accuracy}')
    print(f'Precision score Female: {unt_precision_F}')
    print(f'Precision score Male: {unt_precision_M}')
    print(f'Recall score Female: {unt_recall_F}')
    print(f'Recall score Male: {unt_recall_M}')
    print(f'F1 score: {unt_f1}')
    print(f'Cohen kappa score: {unt_cohen_kappa}')


    """ Tuning with GridSearch """
    
    # Hyperparameters that could be useful to tune

    n_neigh = list(range(1, 20))
    metric =  ['minkowski', 'euclidean', 'manhattan']
    weights = ['uniform', 'distance']

    hyperpara = dict(n_neighbors=n_neigh, weights = weights, metric = metric)

    gs = skl_ms.GridSearchCV(skl_nb.KNeighborsClassifier(), hyperpara, cv=10, refit='balanced_accuracy' , verbose=1, n_jobs=-1)     

    tuned_n = np.zeros(10)

    # See if the tuned parameters are the same for each iteration 

    for i in np.arange(10):
        X_train, _, Y_train, _ = skl_ms.train_test_split(X, Y, test_size=0.3) 
        tuned_model = gs.fit(X_train, Y_train)    
        print(f'Tuned model parameters: {tuned_model.best_params_}')
        dictionary_para = tuned_model.best_params_
        tuned_n[i] += dictionary_para['n_neighbors']
    
    avg_tuned_n = np.mean(tuned_n)

    print(f'Tuned nearest neighbors: {avg_tuned_n}')
    

    """ Evaluate the results for the tuned model"""
    
    accuracy, bal_accuracy, precision_F, precision_M, recall_F, recall_M, f1, cohen_kappa = evaluation_cross_val(X, Y, n_folds=10, tuned=True)

    print('Tuned model:')
    print(f'Accuracy score: {accuracy}')
    print(f'Balanced accuracy: {bal_accuracy}')
    print(f'Precision score Female: {precision_F}')
    print(f'Precision score Male: {precision_M}')
    print(f'Recall score Female: {recall_F}')
    print(f'Recall score Male: {recall_M}')
    print(f'F1 score: {f1}')
    print(f'Cohen kappa score: {cohen_kappa}')

##########################################################
## RUN CODE

if __name__ == '__main__':
    main()