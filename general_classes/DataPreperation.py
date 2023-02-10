### IMPORTS ###

import numpy as np
import pandas as pd
import sys
import os


### CHECKING FOLDERS ###

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/general_classes", "")

### GLOBALS ###

clear = lambda : os.system("cls")


class DataPreperation():
    def __init__(self, path_data):

        try:
            if sys.platform == "darwin": # for macOS
                self.data = pd.read_csv(os.path.join(dirname, path_data)) 
            else:
                self.data = pd.read_csv(path_data)
        except:
            print("FileNotFoundError: [Errno 2] No such file or directory")

        self.x_length = self.data.shape[0]
        self.y_length = self.data.shape[1]


    def create_data_sets(self):
        train = self.data.sample(frac = .7, random_state=10)
        test = self.data.drop(train.index)

        # CLEAR COLUMNS AND PREPARE DATA

        return train, test 

        
        

