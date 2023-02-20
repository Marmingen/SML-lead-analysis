import sys
sys.path.append('.')

import sklearn

from general_classes import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn import preprocessing


import numpy as np

from imblearn.over_sampling import SMOTE


def main():
    dp = DataPreparation("./data/train.csv", clean=False)
        
    X_train, X_test, Y_train, Y_test = dp.get_sets()
    
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((Y_train, Y_test))
    
    # normal
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    
    y_predict = np.ones(Y_test.shape)
    
    print(y_predict)
    
    perf = Performance(y_predict, Y_test)
    
    dic = get_dict()
    
    perf.combination(dic)
    
    print_combination(dic)
    
    
    
    
    
if __name__ == "__main__":
    main()