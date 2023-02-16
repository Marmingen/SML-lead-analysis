# Analysis the data for the given data set to evaluate if certain parameters have more impact than others. 

### IMPORT ###

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.model_selection as skl_ms
import sklearn.feature_selection as skl_fs
import sklearn.preprocessing as skl_pp
import os

### FUNCTIONS ###

def info_csv(data_frame):
    data_frame.info()
    print(data_frame.describe)
    print(data_frame.isnull().sum())

def normalize(X):
        return (X - X.min())/(X.max() - X.min())

### main ###

def main():

    # Import training data to analyse

    df = pd.read_csv('./data/train.csv')

    #info_csv(df)                                # No zero data points, OK

    ### IMPORT DATA ###

    # Distibution of 'Male' and 'Female' in Lead
    plt.figure(1)
    sns.countplot(x = 'Lead', data = df)         # Highly unbalanced -> have to balance the data

    # Create input, X, and output, Y, vectors. Output = 'Lead'
    X = df.drop(columns='Lead')
    Y = df['Lead']

    # Normalize the input data from 0 to 1. Models are getting benefits from it
    normalize_X = normalize(X)

    # Combine it to a single data frame with columns representing the features and rows representing the observations -> 13506 rows 
    # Reshape it to make it more manageable for the computer processing. Easier for visualisation.

    data = pd.concat([normalize_X, Y], axis=1)
    # data_1 = data.drop(columns=['Difference in words lead and co-lead','Number of male actors', 'Year', 'Number of female actors', \
    # 'Gross','Mean Age Male' ,'Mean Age Female','Age Lead', 'Age Co-Lead', 'Number of words lead'])
    # print(data_1)
    data = pd.melt(data, id_vars='Lead', var_name='Features', value_name='Value')

    print(data)

    # print(data)

    ### VISUALIZE ###

    plt.figure(2)
    # sns.swarmplot(x= 'Features', y= 'Value', hue= 'Lead', data= data, size=1.4, linewidth=2)
    sns.heatmap(X.corr(), annot=True)

    ### SELECTION ###

    X_train, X_val, Y_train, Y_val = skl_ms.train_test_split(normalize_X, Y, random_state=10)

    # Find the # best features 

    best_ft = skl_fs.SelectKBest(skl_fs.chi2, k=9).fit(X_train, Y_train)

    print(f'Score: {best_ft.scores_}')
    
    best_ft = skl_fs.SelectKBest(skl_fs.f_classif, k=9).fit(X_train, Y_train)

    print(f'Score: {best_ft.scores_}')
    print(f'Columns: {X_train.columns}')

    # Preprocessing
    # X_train_2 = best_ft.transform(X_train)
    # X_val_2 = best_ft.transform(X_val)

    # X = pd.DataFrame(X_train_2)
    # X_train = skl_pp.StandardScaler().fit(X_train_2).transform(X_train_2.astype(float))
    # X_1 = pd.DataFrame(X_train)
    # X_val = skl_pp.StandardScaler().fit(X_val_2).transform(X_val_2.astype(float))
    # print(X, X_1)
    plt.show()

if __name__ == '__main__':
    main()