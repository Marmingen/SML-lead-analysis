#################################################
# this class is used by boosting and random forsest methods
#################################################
## IMPORTS

#################################################
## CLASS
class Tree():
    
    class Node:
        def __init__(self, limit, left=None, right=None):
            self.limit = limit
            self.left = left
            self.right = right
            self.leaf = True

        def __iter__(self):
            if self.left:
                yield from self.left
            yield self.key
            if self.right:
                yield from self.right
                
    def __init__(self, train_data, name, p = 0, max_depth=5, root=None):
        self.root = root
        self.train_data = train_data
        self.p = p
        
    def __str__(self):
        return f"{self.name} | depth: {self.depth}"
    
    def __repr__(self):
        return f"{self.name}"
    
    def split(self, R):
        return self._split(R)
    
    def _split(self, R, T):
        
        # if depth is reached or split is perfect, terminate branch
        
        if self.max_depth == 4:
            return None 
        else:
            pass
        
        pass
        
    

    
    