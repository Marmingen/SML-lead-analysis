############################################################
#
#
#
############################################################
## IMPORTS

import sys
sys.path.append(str(sys.path[0][:-14]))
from sys import platform

import general_methods as gm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statistics as stat
import os
from scipy.stats import t

dirname = os.getcwd()
dirname = dirname.replace("/stat_analysis", "")


############################################################
############################################################
## GLOBALS

# function that clears the terminal
clear = lambda : os.system('cls')

# converts the strings into binaries. 1 for "Female", 0 for everything else
gender_to_bin = lambda g : int(bool(g == "Female"))

bar = "************************************************************"

############################################################
############################################################
## FUNCTIONS         

def bin_gender(data):
    """
    
    """
    
    genders = [gender_to_bin(entry) for entry in data.iloc[:,-1]]
    
    male = genders.count(0)
    
    female = genders.count(1)
    
    mu = stat.mean(genders)
    
    sigma = stat.stdev(genders)
    
    N = stat.NormalDist(mu, sigma)
    
    print(f"{'female leads: ':{'.'}<30}{f' {female}':{'.'}>30}")
    print(f"{'male leads: ':{'.'}<30}{f' {male}':{'.'}>30}")
    print(f"{'total samples: ':{'.'}<30}{f' {male + female}':{'.'}>30}")
    print(bar, "\n")
    print("APPROXIMATION TO NORMAL DISTRIBUTION")
    print(bar)
    
    meanstr = f"mean [\u03BC]: "
    stdstr = f"standard deviation [\u03C3]: "
    invar = "P(gender \u2208 [0.5,1]): " 
    
    print(f"{meanstr:{'.'}<30}{f' {round(mu, 3)}':{'.'}>30}")
    print(f"{stdstr:{'.'}<30}{f' {round(sigma, 3)}':{'.'}>30}")
    print(f"{invar:{'.'}<30}{f' {round(1-N.cdf(0.5),3)}':{'.'}>30}")
    print(bar)
    
    
    x = list(np.linspace(0,1,200))
    y = [N.pdf(xt) for xt in x]
    
    # PLOTTING
    ########################################################
    plt.figure(3)
    plt.plot(x,y)
    
    plt.axvline(x = 0.5, color = 'r', linestyle="--")
    plt.grid()
    plt.xticks([0 + 0.1*k for k in range(11)])
    plt.xlabel("Binary gender [Male,Female]=[0,1]")
    plt.ylabel("Probability density")
    plt.title("Patriarchy            Equality            Matriarchy")
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/normal.png"))
    ########################################################
    

    # LÄGG TILL KONFIDENSINTERVALL

    return [male, female]


def time_gender(data):
    
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
    theta1 = gm.lin_reg(x_years, frac)[0]
    x1 = [x_years[0], x_years[-1]]
    y1 = [theta1[0] + x1[0]*theta1[1], theta1[0]+x1[1]*theta1[1]]
    
    # using all of the data after 1980 (due to the few samples from earlier dates)
    theta2 = gm.lin_reg(x_years[15:], frac[15:])[0]
    x2 = [x_years[15], x_years[-1]]
    y2 = [theta2[0] + x2[0]*theta2[1], theta2[0]+x2[1]*theta2[1]]
    
    # PLOTTING
    ########################################################

    # the number of male and female leads
    plt.figure(0)
    plt.plot(x_years, males, "b", x_years, females, "r")
    
    plt.title("Amount of movies with leads of binary gender over time")
    plt.grid()
    plt.legend(["Male", "Female"])
    plt.xlabel("Year [1939-2015]")
    plt.ylabel("Amount of movies")
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/numbers.png"))
    
    # the fraction of female leads, and two linear regressions
    plt.figure(1)
    plt.plot(x_years, frac, "--k", x1, y1, "r", x2, y2, "b")
    
    plt.legend(["Data", "1939-2015", "1980-2015"])
    plt.grid()
    plt.title("Fraction of female leads over time")
    plt.xlabel("Year [1939-2015]")
    plt.ylabel("Fraction of female leads [%]")
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/fraction.png"))
    
    # the total number of analyzed movies per year
    plt.figure(2)
    plt.plot(x_years, amt)
    
    plt.grid()
    plt.title("Total amount of analyzed movies over time")
    plt.xlabel("Year [1939-2015]")
    plt.ylabel("Total amount of movies in data")
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/amount.png"))
    
    
def gross_gender(data):
    
    gross_data = data[["Gross", "Lead"]].values.tolist()
    
    male = sorted([entry[0] for entry in gross_data if entry[1] == "Male"])
    
    female = sorted([entry[0] for entry in gross_data if entry[1] == "Female"])
    
    mu_male = stat.mean(male)
    mu_female = stat.mean(female)
    
    sigma_male = stat.stdev(male)
    
    sigma_female = stat.stdev(female)
    
    N_male = stat.NormalDist(mu_male, sigma_male)
    N_female = stat.NormalDist(mu_female, sigma_female)

    x_male = np.linspace(0,male[-1],200)
    y_male = [N_male.pdf(x) for x in x_male]
    
    x_female = np.linspace(0,female[-1],200)
    y_female = [N_female.pdf(x) for x in x_female]
    
    # PLOTTING    
    ########################################################
    plt.figure(4)
    plt.plot(x_female, y_female, '--r', label='Female')
    plt.plot(x_male, y_male, '--b', label='Male')
    plt.axvline(x = mu_female, color = 'r', linestyle=":")
    plt.axvline(x = mu_male, color = 'b', linestyle=":")
    plt.title("Normal distribution of the grossing based on lead gender")
    plt.xlabel("Grossing")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(dirname, "stat_analysis/graphs/grossing.png"))

    ########################################################
    
    fmu_str = "female mean [\u03BC]: "
    fsi_str = "female strd deviation [\u03C3]: "
    mmu_str = "male mean [\u03BC]: "
    msi_str = "male strd deviation [\u03C3]: "
    
    print("\nGROSSING BY GENDER")
    print(bar)
    print(f"{fmu_str:{'.'}<30}{f' {round(mu_female,2)}':{'.'}>30}")
    print(f"{fsi_str:{'.'}<30}{f' {round(sigma_female,2)}':{'.'}>30}\n")
    print(f"{mmu_str:{'.'}<30}{f' {round(mu_male,2)}':{'.'}>30}")
    print(f"{msi_str:{'.'}<30}{f' {round(sigma_male,2)}':{'.'}>30}")
    two_sample_t_test(y_female, y_male)
    print(bar)


def two_sample_t_test(data2, data1):
    """
    Calculates p-value for two sample t test
    """

    mean1, mean2 = np.mean(data1), np.mean(data2) # mean
    std1, std2 = np.std(data1), np.std(data2) # standard deviation
    n1, n2 = len(data1), len(data2) # sample size
    dof1, dof2 = n1 - 1, n2 - 1 # degree of freedom

    se1, se2 = std1/np.sqrt(n1), std2/np.sqrt(n2) # standard errors
    sed = np.sqrt(se1**2 + se2**2) # standard error on the difference between samples
     
    t_stat = (mean1 - mean2) / sed # t-statistic

    df = dof1 + dof2 # total degree of freedom
    alpha = 0.05 # confidence level
    cr = np.abs(t.ppf((1-alpha)/2,df)) # critical region
    
    p_value = (1 - t.cdf(abs(t_stat), df)) * 2 # p-value

    
    print("\n\nTWO SAMPLE T-TEST\n")
    print(f"p-value: \t {p_value}\n")

    if p_value <= alpha:
        print("Reject null hypotheses at 5 % level: \t Distributions are not the same")
    else:
        print("Cant reject null hypotheses at 5 % level: \t Distributions could be the same")

    print(f"mean1:{mean1} mean2:{mean2} std1:{std1} std2:{std2}")
    return p_value




def gross_lines(data):
    
    lines_data_F = data[["Gross", "Number words female"]].sort_values(by="Number words female").values.tolist()
    lines_data_M = data[["Gross", "Number words male"]].sort_values(by="Number words male").values.tolist()
    
    gross_F = [entry[0] for entry in lines_data_F]
    lines_F = [entry[1] for entry in lines_data_F]
    
    gross_M = [entry[0] for entry in lines_data_M]
    lines_M = [entry[1] for entry in lines_data_M]
    
    mu = stat.mean(gross_F)
    mu2 = stat.mean(gross_M)
    
    plt.figure(6)
    plt.grid()
    plt.scatter(lines_F, gross_F)
    plt.xlabel("Number of lines by female characters")
    plt.ylabel("Grossing")
    plt.show()
    
    yvals_F = gm.log_reg(lines_F, gross_F)
    yvals2_M = gm.log_reg(lines_M, gross_M)
    
    plt.figure(7)
    plt.plot(lines_F, yvals_F)
    plt.grid()
    plt.show()


    plt.figure(8)
    plt.grid()
    plt.scatter(lines_M, gross_M)
    plt.xlabel("Number of lines by male characters")
    plt.ylabel("Grossing")
    plt.show()

    
    # print(lines_data)
    

    

def grossing_age(data):
    gross_data = data[["Gross", "Mean Age Male", "Mean Age Female"]].values.tolist()
    
    gross = [entry[0] for entry in gross_data]
    
    male_data = [entry[1] for entry in gross_data]
    
    female_data = [entry[2] for entry in gross_data]
    
    mu_male = stat.mean(male_data)
    mu_female = stat.mean(female_data)
    
    sigma_male = stat.stdev(male_data)
    
    sigma_female = stat.stdev(female_data)
    
    N_male = stat.NormalDist(mu_male, sigma_male)
    N_female = stat.NormalDist(mu_female, sigma_female)

    x_male = np.linspace(0,male_data[-1],200)
    y_male = [N_male.pdf(x) for x in x_male]
    
    x_female = np.linspace(0,female_data[-1],200)
    y_female = [N_female.pdf(x) for x in x_female]
    
    # PLOTTING    
    ########################################################
    plt.figure(4)
    
    plt.scatter(gross, male_data)
    plt.scatter(gross, female_data)
    
    plt.show()
    # plt.plot(x_female, y_female, '--r', label='Female')
    # plt.plot(x_male, y_male, '--b', label='Male')
    # plt.axvline(x = mu_female, color = 'r', linestyle=":")
    # plt.axvline(x = mu_male, color = 'b', linestyle=":")
    # plt.title("Normal distribution of the grossing based on lead gender")
    # plt.xlabel("Grossing")
    # plt.legend()
    # plt.grid()
    # plt.savefig("stat_analysis/graphs/grossing.png")
    ########################################################
    
    fmu_str = "female mean [\u03BC]: "
    fsi_str = "female strd deviation [\u03C3]: "
    mmu_str = "male mean [\u03BC]: "
    msi_str = "male strd deviation [\u03C3]: "
    
    print("\nGROSSING BY GENDER")
    print(bar)
    print(f"{fmu_str:{'.'}<30}{f' {round(mu_female,2)}':{'.'}>30}")
    print(f"{fsi_str:{'.'}<30}{f' {round(sigma_female,2)}':{'.'}>30}\n")
    print(f"{mmu_str:{'.'}<30}{f' {round(mu_male,2)}':{'.'}>30}")
    print(f"{msi_str:{'.'}<30}{f' {round(sigma_male,2)}':{'.'}>30}")
    print(bar)

############################################################
############################################################
## MAIN
          
def main():
    if platform == "darwin": # check for mac os
        training_data = pd.read_csv(os.path.join(dirname, "data/train.csv"))

    else:
        training_data = pd.read_csv("data/train.csv") 

    print("STATISTICAL ANALYSIS OF THE TRAINING DATA")
    print(bar)
    
    bin_gender(training_data)
    
    time_gender(training_data)

    gross_gender(training_data)
    
    # gross_lines(training_data)
    
    grossing_age(training_data)

############################################################
############################################################
## RUN CODE
    
if __name__ == "__main__":
    main()
