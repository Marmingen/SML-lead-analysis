############################################################
## IMPORTS

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics as stat
import os

from sys import platform
from scipy.stats import t

############################################################
## FIXING PATH

sys.path.append(str(sys.path[0][:-14]))
dirname = os.getcwd()
dirname = dirname.replace("/stat_analysis", "")

############################################################
## LOCAL PACKAGES

import general_methods as gm

############################################################
## GLOBALS

# function that clears the terminal
clear = lambda : os.system('cls')

# converts the strings into binaries. 1 for "Female", 0 for everything else
gender_to_bin = lambda g : int(bool(g == "Female"))

bar = "************************************************************"

############################################################
## FUNCTIONS

def bin_roles(data):
    
    numb_actors = data[["Number of male actors", "Number of female actors"]].values.tolist()
    
    actors_M = [entry[0] for entry in numb_actors]
    numb_M = sum(actors_M)
    actors_F = [entry[1] for entry in numb_actors]
    numb_F = sum(actors_F)
    
    roles = []
    
    for M, F in zip(actors_M, actors_F):
        roles += [-1]*M + [1]*F
    
    mu = stat.mean(roles)
    
    sigma = stat.stdev(roles)
    
    N = stat.NormalDist(mu, sigma)
    
    print(f"{'female roles: ':{'.'}<30}{f' {numb_F}':{'.'}>30}")
    print(f"{'male roles: ':{'.'}<30}{f' {numb_M}':{'.'}>30}")
    print(f"{'total roles: ':{'.'}<30}{f' {numb_M + numb_F}':{'.'}>30}")
    print(bar, "\n")
    print("APPROXIMATION TO NORMAL DISTRIBUTION")
    print(bar)
    
    meanstr = f"mean [\u03BC]: "
    stdstr = f"standard deviation [\u03C3]: "
    invar = "P(gender \u2208 [0,1]): " 
    
    print(f"{meanstr:{'.'}<30}{f' {round(mu, 3)}':{'.'}>30}")
    print(f"{stdstr:{'.'}<30}{f' {round(sigma, 3)}':{'.'}>30}")
    print(f"{invar:{'.'}<30}{f' {round(1-N.cdf(0),3)}':{'.'}>30}")
    print(bar)
    
    x = list(np.linspace(-1,1,200))
    y = [N.pdf(xt) for xt in x]
    
    # PLOTTING
    ########################################################
    plt.figure(1)
    plt.plot(x,y)
    plt.axvline(x = 0, color = 'r', linestyle="--")
    plt.grid()
    plt.xticks([-1 + 0.2*k for k in range(11)])
    plt.xlabel("Binary gender [Male,Female]=[-1,1]")
    plt.ylabel("Probability density")
    plt.title("Patriarchy            Equality            Matriarchy")
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/bin-roles.png"))
    ########################################################
    
def main():
    
    if platform == "darwin": # check for mac os
        training_data = pd.read_csv(os.path.join(dirname, "data/train.csv"))

    else:
        training_data = pd.read_csv("data/train.csv") 

    print("STATISTICAL ANALYSIS OF THE TRAINING DATA")
    print("")
    print("### MAJOR ROLE BY GENDER ###")
    print(bar)
    print(bar)
    
    bin_roles(training_data)
    print(bar, "\n")
    print("### WORDS SPOKEN AND GROSSING ###")
    
    
if __name__ == "__main__":
    main()