############################################################



############################################################
## IMPORTS

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics as stat
import os

############################################################
## GLOBALS

# function that clears the terminal
clear = lambda : os.system('cls')

bar = "************************************************************"

############################################################
## FUNCTIONS

def read_data(filename):
    data = pd.read_csv(filename)
        
    return data
         
         
def bin_gender(data):
    """
    
    """
    
    genders_s = data.iloc[:,-1].values.tolist()
    
    male = genders_s.count("Male")
    
    female = genders_s.count("Female")
    
    return [male, female]


def lin_reg(xdata, ydata):
    
    ones = np.ones(len(xdata))
    
    Xtemp = np.array(xdata)
    
    Xt = np.vstack((ones, Xtemp))   # X'
    
    X = np.transpose(Xt)
    
    Y = np.transpose(np.array(ydata))
    
    Z = np.linalg.inv(Xt@X)
    
    # (X'X)**(-1)X'Y
    return Z@Xt@Y

def time_gender(data):
    
    # converts the strings into binaries. 1 for "Female", 0 for everything else
    gender_to_bin = lambda g : int(bool(g == "Female"))
    
    years = data[["Year", "Lead"]].sort_values(by="Year").values.tolist()
    
    # year: [male, female]
    dict_years = {}
    
    # counting the amount of movies with Male or Female lead
    for entry in years:
        gender_index = gender_to_bin(entry[1])
        try:
            dict_years[entry[0]][gender_index] += 1
        except:
            dict_years[entry[0]] = [0,0]
            dict_years[entry[0]][gender_index] = 1
    
    # makes two Y-lists
    males = [entry[0] for entry in dict_years.values()]
    females = [entry[1] for entry in dict_years.values()]
    
    # calculates the fraction of female leads in percent
    frac = [100*entry[1]/sum(entry) for entry in dict_years.values()]
    
    # calculates the total number of movies per year
    amt = [sum(entry) for entry in dict_years.values()]
    
    # sets a list of the x-values
    x_years = list(dict_years.keys())
    
    # linear regression for the fraction of female leads
    # using all of the data
    theta1 = lin_reg(x_years, frac)
    x1 = [x_years[0], x_years[-1]]
    y1 = [theta1[0] + x1[0]*theta1[1], theta1[0]+x1[1]*theta1[1]]
    
    # using all of the data after 1980 (due to the few samples from earlier dates)
    theta2 = lin_reg(x_years[15:], frac[15:])
    x2 = [x_years[15], x_years[-1]]
    y2 = [theta2[0] + x2[0]*theta2[1], theta2[0]+x2[1]*theta2[1]]
    
    # PLOTTING    
    ##########################################################

    # the number of male and female leads
    plt.figure(0)
    plt.plot(x_years, males, "b", x_years, females, "r")
    
    plt.title("Amount of movies with leads of binary gender over time")
    plt.grid()
    plt.legend(["Male", "Female"])
    plt.xlabel("Year [1939-2015]")
    plt.ylabel("Amount of movies")
    plt.savefig("stat_analysis/graphs/numbers.png")
    
    # the fraction of female leads, and two linear regressions   
    plt.figure(1)
    plt.plot(x_years, frac, "--k", x1, y1, "r", x2, y2, "b")
    
    plt.legend(["Data", "1939-2015", "1980-2015"])
    plt.grid()
    plt.title("Fraction of female leads over time")
    plt.xlabel("Year [1939-2015]")
    plt.ylabel("Fraction of female leads [%]")
    plt.savefig("stat_analysis/graphs/fraction.png")
    
    # the total number of analyzed movies per year
    plt.figure(2)
    plt.plot(x_years, amt)
    
    plt.grid()
    plt.title("Total amount of analyzed movies over time")
    plt.xlabel("Year [1939-2015]")
    plt.ylabel("Total amount of movies in data")
    plt.savefig("stat_analysis/graphs/amount.png")
    
############################################################
## MAIN
          
def main():
    
    training_data = pd.read_csv("data/train.csv")
    test_data = pd.read_csv("data/test.csv")
    
    # print(test_data[0])
    
    
    # print(test_data.iloc[:,0])
    
    bin_gender(training_data)
    
    time_gender(training_data)
    
############################################################
## RUN CODE
    
if __name__ == "__main__":
    main()
