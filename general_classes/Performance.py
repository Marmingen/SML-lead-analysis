import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, cohen_kappa_score, roc_curve


class Performance():
    def __init__(self, Y_pred, Y_true):
        self.Y_pred = Y_pred
        self.Y_true = Y_true
        
        self.binary_label = [-1,1]
        self.normal_label = ["Female", "Male"]
    
    def accuracy(self):
        return np.sum(self.Y_true == self.Y_pred) / len(self.Y_true)
    
    def precision(self, gender=False):
        male_precision = precision_score(self.Y_true, self.Y_pred, pos_label="Male" if gender else 1)
        female_precision = precision_score(self.Y_true, self.Y_pred, pos_label="Female" if gender else -1)
        
        return (male_precision, female_precision)
    
    def recall(self, gender=False):
        male_recall = recall_score(self.Y_true, self.Y_pred, pos_label="Male" if gender else 1)
        female_recall = recall_score(self.Y_true, self.Y_pred, pos_label="Female" if gender else -1)
        
        return (male_recall, female_recall)
    
    def confusion(self):
        return confusion_matrix(self.Y_true, self.Y_pred)
    
    def f1(self, gender=False):
        male_f1 = f1_score(self.Y_true, self.Y_pred, pos_label="Male" if gender else 1)
        female_f1 = f1_score(self.Y_true, self.Y_pred, pos_label="Female" if gender else -1)
    
    def cohen(self):
        return cohen_kappa_score(self.Y_true, self.Y_pred)
    
    def roc(self, gender=False):
        m_fpr, m_tpr, m_thres = roc_curve(self.Y_true, self.Y_pred, pos_label="Male" if gender else 1)
        f_fpr, f_tpr, f_thres = roc_curve(self.Y_true, self.Y_pred, pos_label="Female" if gender else -1)
        
        return [[m_fpr, m_tpr, m_thres], [f_fpr, f_tpr, f_thres]]