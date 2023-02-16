import sys
sys.path.append('.')

from general_classes import *

def main():
    
    dp = DataPreperation("./data/train.csv")
    
    train, test = dp.create_data_sets()
    
    T1 = Tree(train, "T1", 3)
    
    T1.train()

if __name__ == "__main__":
    main()