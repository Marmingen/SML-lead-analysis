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

### GLOBALS ###
clear = lambda : os.system("cls")


### MAIN ###

def main():
    pass



if __name__ == "__main__":
    main()


