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
        def __init__(self, train_data, left=None, right=None, depth = 1):
            self.train_data = train_data
            self.left = left
            self.right = right
            self.depth = depth
            
            self.leaf = False
            self._majority()
        
        def _majority(self):
            n_F = len(self.train_data[self.train_data["Lead"] == "Female"])
            n_M = len(self.train_data[self.train_data["Lead"] == "Male"])
            
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
                
    def __init__(self, train_data, name="unnamed", max_depth=5, disp=True):
        self.train_data = train_data
        self.name = name
        self.max_depth = max_depth
        self.disp = disp
        
        self.root = self.Node(train_data)
        
    def __str__(self):
        return f"{self.name} | depth: {0}"
    
    def __repr__(self):
        return f"{self.name}"
    
    def train(self):
        self.total = 0
        self._split(self.root)
    
    def _split(self, R):
        
        def calc_Q(pi_M, pi_F):
            try:
                return -(pi_M*log(pi_M) + pi_F*log(pi_F))
            except ValueError:
                return 0
        
        def calc_pi(T_set, n, gender):
            try:
                return T_set.Lead.value_counts()[gender]/n
            except KeyError:
                return 0
        
        # if depth is reached or split is perfect, leaf
        
        if R.depth == self.max_depth or R.leaf:
            R.leaf = True
        else:
            # initializing variables
            T1_best = None
            T2_best = None
            lowest_amin = 100
            best_set = []
            
            T = R.train_data
            
            for x_i in T:
                if x_i == "Lead":
                    continue
                
                if self.disp:
                    self.total += 1
                    print(f'\rTotal variables checked: {self.total}', end = "\r")
                
                for s in T[x_i]:

                    T1 = T[T[x_i] < s]
                    T2 = T[T[x_i] >= s]
                    
                    n1 = len(T1.index)
                    n2 = len(T2.index)
                    
                    if n1*n2 == 0:
                        continue
                    
                    pi_1M = calc_pi(T1, n1, "Male")
                    pi_1F = calc_pi(T1, n1, "Female")
                    
                    pi_2M = calc_pi(T2, n2, "Male")
                    pi_2F = calc_pi(T2, n2, "Female")
                    
                    Q1 = calc_Q(pi_1M, pi_1F)
                    Q2 = calc_Q(pi_2M, pi_2F)
                    
                    amin = np.argmin(n1*Q1 + n2+Q2)
                    
                    if amin < lowest_amin:
                        lowest_amin = amin
                        best_set = [x_i, s]
                        
                        T1_best = T1
                        T2_best = T2
            
            R.limit = best_set
            
            R.left = self.Node(T1_best, depth=R.depth+1)
            self._split(R.left)
            
            R.right = self.Node(T2_best, depth=R.depth+1)
            self._split(R.right)
    
    def predict(self, data):
        
        R = self.root
        
        while not R.leaf:
            val = data[R.limit[0]]
            lim = R.limit[1]
            
            if val < lim:
                R = R.left
            else:
                R = R.right
            
        return R.probable