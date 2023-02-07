### IMPORTS ###
import numpy as np
import sys
import os

from DecisionStump import DecisionStump
from AdaBoost import AdaBoost

### CHECKING FOLDERS ###
sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting", "")

sys.path.insert(1,os.path.join(dirname, "general_classes"))

from DataPreperation import DataPreperation


### GLOBALS ###
clear = lambda : os.system("cls")


### MAIN ###

def main():
    path_data = dirname + "/data/train.csv"
    DataPrep = DataPreperation(path_data)
    train_set, test_set = DataPrep.create_data_sets()
    print(train_set)
    print(test_set)




if __name__ == "__main__":
    main()


