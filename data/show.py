############################################################
## IMPORTS

import pandas as pd

bar = "************************************************************"


def show_data():
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

if __name__ == "__main__":
    show_data()