### IMPORTS ###

import numpy as np
import pandas as pd
import sys
import os

### CHECKING FOLDERS ###

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/models/boosting", "")
sys.path.insert(1, os.path.join(dirname, "general_classes"))

from Tree import Tree

class GradientBoosting():
    def __init__(self):
        pass


