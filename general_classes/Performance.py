import numpy as np


class Performance():
    def __init__(self, Y_pred, Y_true):
        self.Y_pred = Y_pred
        self.Y_true = Y_true

    def accuracy(self):
