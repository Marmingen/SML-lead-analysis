############################################################
## IMPORTS

import os
import pandas as pd

############################################################
## GLOBALS

clear = lambda : os.system('cls')

bar = "************************************************************"

############################################################
## FUNCTIONS

def show_data():
    clear()
    training_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    
    print("training data")
    print(bar)
    print(training_data.info())
    print(bar)
    print(training_data.head())
    print(bar, "\n")
    
    print("test data")
    print(bar)
    print(test_data.info())
    print(bar)
    print(test_data.head())
    print(bar)
    input("press enter to go back")

############################################################
## RUN CODE

if __name__ == "__main__":
    show_data()