#################################################
# this class is used by boosting and random forsest methods
#################################################
## IMPORTS

from math import log
import numpy as np

#################################################
## TREE CLASS
class Tree():
    
    # Node class for handling the split info storage and predictions
    class Node:
        def __init__(self, train_data, left=None, right=None):
            self.train_data = train_data
            self.left = left
            self.right = right
            self.leaf = False
            self._majority()
        
        def _majority(self):
            n_F = self.train_data[self.train_data["Lead"] == "Female"]
            n_M = self.train_data[self.train_data["Lead"] == "Male"]
            
            # bias towards male leads due to higher probability
            if n_F>n_M:
                self.probable = "Female"
            else:
                self.probable = "Male"
            
            # perfect split, only contains one class
            if n_F*n_M == 0:
                self.leaf = True
            
        # def __iter__(self):
        #     if self.left:
        #         yield from self.left
        #     yield self.key
        #     if self.right:
        #         yield from self.right
                
    def __init__(self, train_data, name="unnamed", max_depth=5):
        self.train_data = train_data
        self.name = name
        self.max_depth = max_depth
        
        self.depth = 0
        self.root = self.Node(train_data)
        
    def __str__(self):
        return f"{self.name} | depth: {self.depth}"
    
    def __repr__(self):
        return f"{self.name}"
    
    def train(self):
        self._split(self.root)
    
    def _split(self, R):
        
        def calc_Q(pi_M, pi_F):
            return -(pi_M*log(pi_M) + pi_F*log(pi_F))
        
        self.depth += 1
        
        # if depth is reached or split is perfect, leaf
        
        if self.depth == self.max_depth or R.leaf:
            R.leaf = True
        else:
            
            T1_best = None
            T2_best = None
            
            lowest_amin = 100
            best_set = []
            
            T = R.train_data
            
            for x_i in T:                
                for s in x_i:
                    
                    T1 = T[T[x_i] < s]
                    T2 = T[T[x_i] >= s]
                    
                    n1 = len(T1.index)
                    n2 = len(T2.index)
                    
                    if n1*n2 == 0:
                        continue
                    
                    pi_1M = T1.Lead.value_counts()["Male"]/n1
                    pi_1F = T1.Lead.value_counts()["Female"]/n1
                    
                    pi_2M = T2.Lead.value_counts()["Male"]/n2
                    pi_2F = T2.Lead.value_counts()["Female"]/n2
                    
                    Q1 = calc_Q(pi_1M, pi_1F)
                    Q2 = calc_Q(pi_2M, pi_2F)
                    
                    amin = np.argmin(n1*Q1 + n2+Q2)
                    
                    if amin < lowest_amin:
                        lowest_amin = amin
                        best_set = [x_i, s]
                        
                        T1_best = T1
                        T2_best = T2
            
            R.limit = best_set
            
            R.left = self.Node(T1_best)
            self._split(R.left)
            
            R.right = self.Node(T2_best)
            self._split(R.right)
    

    
    