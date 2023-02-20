##########################################################
# OBSERVE!!
# THIS CLASS IS DEPRECATED AND NOT USED
##########################################################
## IMPORTS

import numpy as np
import os 
import sys

##########################################################
## FIXING PATH 

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/general_classes", "")
##########################################################
## PCA CLASS

class PCA():
    def __init__(self, num_comp):
        """
        
        @params:
            self.num_comp: number of dimensions we want after transformation
        """
        self.num_comp = num_comp
        self.mean = None
        self.comps = None


    def fit(self, X):
        """
        Don't need Y data since this is an unsupervised algorithm
        @params:
            X: training data
        @returns:
            None

        """
        self.mean = np.mean(X, axis=0)
        print(self.mean)
        X = X - self.mean

        # Calculate covariance
        cov = np.cov(X.T)
        
        # Calculate eigenvectors and eigenvalues
        eig_vectors, eig_values = np.linalg.eig(cov)
        eig_vectors = eig_vectors.T 

        # Sort the eigenvectors according to eigenvalues
        idxs = np.argsort(eig_values)[::-1] # Descending order
        eig_values = eig_values[idxs]
        eig_vectors = eig_vectors[idxs]

        self.comps = eig_vectors[:self.num_comp] # Only take n first components

    def transform(self, X):
        """
        @params:
            X: training data
        @returns:
            None
        """

        # Project the data
        X = X - self.mean
        return np.dot(X, self.comps.T)
