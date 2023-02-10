import sys
sys.path.append('.')

from general_classes import *
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import sklearn as skl

import graphviz

def main():
    
    dp = DataPreperation("./data/train.csv")
    
    train, test = dp.create_data_sets()
    
    # T1 = Tree(train, "T1", 3, disp=False)
    
    # T1.train()
    
    X_train = train.drop(columns="Lead")
    y_train = train["Lead"]
    
    X_test = test.drop(columns="Lead")
    y_test = test["Lead"]
    
    model = RandomForestClassifier(n_estimators=100)
    
    # model = tree.DecisionTreeClassifier(max_depth=3)
    model.fit(X=X_train,y=y_train) 
    
    y_predict = model.predict(X_test)
    
    print(skl.metrics.precision_score(y_test, y_predict, pos_label="Male"))
    
    pd.crosstab(y_predict, y_test)
    
    
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