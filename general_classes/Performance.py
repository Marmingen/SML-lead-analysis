import numpy as np
from sklearn.metrics import precision_score, recall_score


class Performance():
    def __init__(self, Y_pred, Y_true):
        self.Y_pred = Y_pred
        self.Y_true = Y_true
        
    def accuracy(self):
        accuracy = np.sum(self.Y_true == self.Y_pred) / len(self.Y_true)
        return accuracy
    
    def precision(self):
        male_precision = precision_score(self.Y_true, self.Y_pred, pos_label="Male")
        female_precision = precision_score(self.Y_true, self.Y_pred, pos_label="Female")
        
        return (male_precision, female_precision)
    
    def recall(self):
        male_recall = recall_score(self.Y_true, self.Y_pred, pos_label="Male")
        female_recall = recall_score(self.Y_true, self.Y_pred, pos_label="Female")
        
        return (male_recall, female_recall)
    