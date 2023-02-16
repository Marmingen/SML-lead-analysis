import sys
sys.path.append('.')

from general_classes import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt

import graphviz

def main():
    
    dp = DataPreparation("./data/train.csv")
        
    X_train, X_test, Y_train, Y_test = dp.get_sets()
    
    model = RandomForestClassifier(n_estimators=100)
    
    model.fit(X=X_train,y=Y_train) 
    
    y_predict = model.predict(X_test)
    
    perf = Performance(y_predict, Y_test)
    
    # dot_data = tree.export_graphviz(model, out_file=None, feature_names=X_train.columns, class_names=model.classes_, filled=True,
    #                                 rounded=True, leaves_parallel=True,proportion=True)
    
    # graph = graphviz.Source(dot_data)
    
    # graph.format = "png"
    # graph.render("graph")

    # l = []
    
    # for row in test.iloc:
    #     prediction = T1.predict(row)
    #     truth = row["Lead"]
        
    #     l += [1 if truth == prediction else 0]
    # print()
    # print(l.count(1)/len(l))
    # print(T1.print())

if __name__ == "__main__":
    main()