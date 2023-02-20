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

bar = "##################################################"


def main():
    
    dp = DataPreparation("./data/train.csv", clean=True)
        
    X_train, X_test, Y_train, Y_test = dp.get_sets()
    
    model = RandomForestClassifier(n_estimators=10)
    
    model.fit(X=X_train,y=Y_train) 
    
    y_predict = model.predict(X_test)
    
    perf = Performance(y_predict, Y_test)
    
    X_train = np.concatenate((X_train, X_test))
    Y_train = np.concatenate((Y_train, Y_test))
    
    # normal
    scaler = preprocessing.MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    
    N = 20
    
    kf = KFold(n_splits=N,random_state=None, shuffle=False)
    
    data = get_dict()
    
    for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
        
        print("Calculating ",f"{round((fold+1)/N*100,2)}%",end="\r")
        
        temp_X = X_train[train_index]
        temp_Y = Y_train[train_index]
        
        sm = SMOTE(k_neighbors=5)
        X_res, Y_res = sm.fit_resample(temp_X,temp_Y)
        
        temp_x_test = X_train[test_index]
        temp_y_test = Y_train[test_index]
        
        temp_X = np.concatenate((temp_X, X_res))
        temp_Y = np.concatenate((temp_Y, Y_res))
        
        model = RandomForestClassifier(n_estimators=100,n_jobs=-1)
        # model = QDA()
        
        model.fit(temp_X, temp_Y)
        y_pred = model.predict(temp_x_test)
        
        perf = Performance(y_pred, temp_y_test)
        
        perf.combination(data)
    
    print("\r\n")
    
    print_combination(data)
    
    # acc = perf.accuracy()
    # bal = perf.balanced_accuracy()
    # prec = perf.precision()
    # rec = perf.recall()
    # conf = perf.confusion()
    # f1 = perf.f1()
    # coh = perf.cohen()
    
    
    # print("performance:")
    # print(bar)
    # print("accuracy:", round(acc,2))
    # print("balanced accuracy:", round(bal,2))
    # print("precision:", round(prec[0],2), round(prec[1],2))
    # print("recall:", round(rec[0],2),round(rec[1],2))
    # # print("confusion:", conf)
    # print("f1-score:", round(f1[0],2),round(f1[1],2))
    # print("cohen:", round(coh,2))
    # print(bar)


if __name__ == "__main__":
    main()