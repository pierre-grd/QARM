import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import scipy.optimize
import scipy.stats as sp


co_2 = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")
market_cap = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Market Cap").dropna()
feuille1 = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Feuille 1 - Group_P")
revenue = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Revenue")
sic = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="SIC")
tt_return_index = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="TT Return Index")

#market_cap = market_cap.merge(sic)
#market_cap = market_cap.merge(sic, how="left")

market_cap_sectors = market_cap.merge(feuille1, how="left", on='ISIN')



#----------------------------------------------------------------------------------------------------------------------
# Question 1 - --------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

#Seperate Data by Sector : Extrapolate 3 most represented and Analyze Mean, Variance,
# skewness, kurtosis, minimum, and maximum.


# Creat List of GIC sectors so as to find top 3

mylist = feuille1['GICSSector'].tolist()


import collections
c = collections.Counter(mylist)


# We now know top 3 sectors are Industrials, Financials and Consumer Discretionary

# Delete every company that is not part of the 3 sectors

# Industrials = market_cap()

"""
market_cap1 = market_cap_sectors.loc[market_cap_sectors['GICSSector'].str.contains('Industrials')]
market_cap2 = market_cap_sectors.loc[market_cap_sectors['GICSSector'].str.contains('Financials')]
market_cap3 = market_cap_sectors.loc[market_cap_sectors['GICSSector'].str.contains('Consumer Discretionary')]
three_sectors = [market_cap1, market_cap2, market_cap3]
aggr_sectors = pd.concat(three_sectors)


market_cap1 = market_cap1.drop(['GICSSector', 'Country', 'Region', 'CommonName'], axis=1)
market_cap2 = market_cap2.drop(['GICSSector', 'Country', 'Region', 'CommonName'], axis=1)
market_cap3 = market_cap3.drop(['GICSSector', 'Country', 'Region', 'CommonName'], axis=1)

market_cap1 = market_cap1.iloc[::,1::]
market_cap2 = market_cap2.iloc[::,1::]
market_cap3 = market_cap3.iloc[::,1::]

market_cap1 = market_cap1.T
market_cap2 = market_cap2.T
market_cap3 = market_cap3.T

market_cap1.index = pd.to_datetime(market_cap1.index)
market_cap2.index = pd.to_datetime(market_cap2.index)
market_cap3.index = pd.to_datetime(market_cap3.index)

# Put into yearly prices

market_cap1 = pd.DataFrame.resample(market_cap1,"Y" )
market_cap2 = pd.DataFrame.resample(market_cap2,"Y" )
market_cap3 = pd.DataFrame.resample(market_cap3,"Y" )

market_cap1 = market_cap1.mean()
market_cap2 = market_cap2.mean()
market_cap3 = market_cap3.mean()

# Sum the columns per year

market_cap1_total = market_cap1.sum(axis=1)
market_cap2_total = market_cap2.sum(axis=1)
market_cap3_total = market_cap3.sum(axis=1)

# Calculate total average returns per year per sector

market_cap1_avgreturn = market_cap1_total / market_cap1_total.shift(1)
market_cap2_avgreturn = market_cap2_total / market_cap2_total.shift(1)
market_cap3_avgreturn = market_cap3_total / market_cap3_total.shift(1)

# Calculate total average all stocks returns

market_cap = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Market Cap").dropna()
market_cap_nafree = market_cap.iloc[1::,2::]
market_cap_nafree = market_cap_nafree.T
market_cap_nafree.index = pd.to_datetime(market_cap_nafree.index)
market_cap_nafree = pd.DataFrame.resample(market_cap_nafree, "Y").mean()
market_cap_nafree = market_cap_nafree.sum(axis=1)
market_cap_nafree_avgreturn = market_cap_nafree / market_cap_nafree.shift(1)


# Mean annualized

market_cap1_avgreturn_mean = market_cap1_avgreturn.mean()
market_cap2_avgreturn_mean = market_cap2_avgreturn.mean()
market_cap3_avgreturn_mean = market_cap3_avgreturn.mean()
markret_cap_nafree_avgreturn_mean = market_cap_nafree_avgreturn.mean()


# Standard deviation annualized

market_cap1_avgreturn_std = market_cap1_avgreturn.std()
market_cap2_avgreturn_std = market_cap2_avgreturn.std()
market_cap3_avgreturn_std = market_cap3_avgreturn.std()
markret_cap_nafree_avgreturn_std = market_cap_nafree_avgreturn.std()


# Skewness

market_cap1_avgreturn_skew = market_cap1_avgreturn.skew()
market_cap2_avgreturn_skew = market_cap2_avgreturn.skew()
market_cap3_avgreturn_skew = market_cap3_avgreturn.skew()
market_cap_nafree_avgreturn_skew = market_cap_nafree_avgreturn.skew()


# Kurtosis

market_cap1_avgreturn_kurt = market_cap1_avgreturn.kurt()
market_cap2_avgreturn_kurt = market_cap2_avgreturn.kurt()
market_cap3_avgreturn_kurt = market_cap3_avgreturn.kurt()
market_cap_nafree_avgreturn_kurt = market_cap_nafree_avgreturn.kurt()


# Minimum

market_cap1_avgreturn_min = market_cap1_avgreturn.min()
market_cap2_avgreturn_min = market_cap2_avgreturn.min()
market_cap3_avgreturn_min = market_cap3_avgreturn.min()
markret_cap_nafree_avgreturn_min = market_cap_nafree_avgreturn.min()



# Maximum

market_cap1_avgreturn_max = market_cap1_avgreturn.max()
market_cap2_avgreturn_max = market_cap2_avgreturn.max()
market_cap3_avgreturn_max = market_cap3_avgreturn.max()
markret_cap_nafree_avgreturn_max = market_cap_nafree_avgreturn.max()

"""


# -----------------------------------------------------------------------------------------------------------------------
# Question 2 -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
"""
market_cap = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Market Cap").dropna()
market_cap_nafree = market_cap.iloc[1::, 2::]


# DATA CLEANING & Montly scaled :

market_cap_nafree = pd.DataFrame.transpose(market_cap_nafree)
market_cap_nafree.index = pd.to_datetime(market_cap_nafree.index)
market_cap_nafree = pd.DataFrame.resample(market_cap_nafree, "M").mean()
# pct_change = market_cap_nafree.pct_change(axis=0)
# pct_change = pct_change.iloc[1:,:]
# pct_change_mean = np.mean(pct_change, axis=0)

stock = market_cap_nafree / market_cap_nafree.shift(1)
stock = stock.iloc[1:, :]
cov_excess = stock.cov()
pct_change_mean = np.mean(stock)

# Create a list of randomized weighting vectors :

portfolio_returns = []
portfolio_volatilities = []
weights_vec = []
# replace 97 by the exact number of companies's stock in the portfolio


#NEGATIVE& POSITIVE WEIGHTS :
for x in range(50000):
    weights = np.random.uniform(-1, 1, 97)
    weights /= np.sum(weights)
    portfolio_returns.append(np.sum(pct_change_mean * weights))
    portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(cov_excess, weights))))
    weights_vec.append(weights)

portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)
weights_vec = np.array(weights_vec)

portfolios_frt = pd.DataFrame({"Return": portfolio_returns, "Volatility": portfolio_volatilities})
portfolios_frt.plot(x="Volatility", y="Return", kind="scatter", color="blue", s=4)
plt.xlabel("Monthly Expected Volatility")
plt.xlim([0.05, 0.3])
plt.ylim([0.95, 1.15])
plt.ylabel("Monthly Expected Return")
plt.show()

#POSITIVE WEIGHTS :
portfolio_returns = []
portfolio_volatilities = []
weights_vec = []
for x in range(10000):
    weights = np.random.uniform(0,1,97)
    weights /= np.sum(weights)
    portfolio_returns.append(np.sum(pct_change_mean * weights))
    portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(cov_excess, weights))))
    weights_vec.append(weights)

portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)
weights_vec = np.array(weights_vec)

portfolios_frt = pd.DataFrame({"Return": portfolio_returns, "Volatility": portfolio_volatilities})
portfolios_frt.plot(x="Volatility", y="Return", kind="scatter", color="blue", s=4)
plt.xlabel("Monthly Expected Volatility")
plt.ylabel("Monthly Expected Return")
plt.show()
"""

# -----------------------------------------------------------------------------------------------------------------------
# Question 3 -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
"""
prtf_mean = []
prtf_cov = []
# Generate x -> Px new samples from the original distribution of mean "pct_change_mean, and variance
# the diagonal of "cov_excess", and compute mean return and cov matrix of the new sample
new_P = []

for i in range (1,50):
  print("Portfolio "+str(i)+"/100 Generated")
  for x in range (275): #replace 275 by the new period count if it is shifted to daily returns
    new_P.append(np.random.normal(pct_change_mean, np.diagonal(cov_excess)))
  new_P=pd.DataFrame(new_P)
  var = new_P.cov()
  mean = new_P.mean(axis=0)
  prtf_mean.append(mean)
  prtf_cov.append(var)
  new_P = []

#We now have 100 sample of normaly distributed returns
# from the original data filled into prtf_mean and prtf_cov

prtf_mean = np.transpose(prtf_mean)
prtf_mean = pd.DataFrame(prtf_mean)

portfolio_returns = []
portfolio_volatilities = []

for i in range(100):
 for x in range(49):
    weights = np.random.random(97)
    weights /= np.sum(weights)
    portfolio_returns.append(np.sum(prtf_mean[x]*weights))
    portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(prtf_cov[x],weights))))


portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities).squeeze()


plt.scatter(portfolio_volatilities,portfolio_returns, s=4, color ="blue")
plt.xlabel("Expected Volatility")
plt.ylabel("Expected Return")
plt.show()
"""




# -----------------------------------------------------------------------------------------------------------------------
# Question 4 -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
"""

min = np.min(portfolio_volatilities)
index_min = np.argmin(portfolio_volatilities)

print("The GMVP has an annualized volatility of " + str(min * 12))
print("GMVP Annualized average return is : " + str((np.mean(portfolio_returns[index_min]) - 1) * 12))

stock_for_mv = weights_vec[index_min] * stock

stock_for_mv = pd.DataFrame.mean(stock_for_mv, axis=1)
print("GMV portfolio MIN ANN Return: " + str(pd.DataFrame.min(stock_for_mv) * 12))
print("GMV portfolio MAX ANN Return : " + str(pd.DataFrame.max(stock_for_mv) * 12))


def var_gaussian(r, level=10, modified=True):
    # compute the Z score assuming it was Gaussian
    z = sp.norm.ppf(level / 100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = sp.skew(r)
        k = sp.kurtosis(r)
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )
    return -(r.mean() + z * np.std(r))


def ES(r, cov, alpha=1):
    CVaR_n = alpha ** -1 * sp.norm.pdf(sp.norm.ppf(alpha)) * cov - r
    return CVaR_n


print("MVP portfolio VaR " + str(var_gaussian(stock_for_mv)))
print("MVP portfolio ES " + str(ES(stock_for_mv.mean(),
                                   cov=portfolio_volatilities[index_min])))

equal_weight = np.full(97, 1 / 97)
EW_returns = np.mean(equal_weight * stock, axis=1)
print("The equally weighted portfolio have an annual volatility of " +
      str(np.sqrt(np.dot(equal_weight.T, np.dot(cov_excess, equal_weight))) * 12))
print("EW portfolio average annual return : " + str((((np.mean(EW_returns)))) * 12))
print("EW portfolio MAX ANN Return : " + str(((np.max(EW_returns))) * 12))
print("EW portfolio MIN ANN Return : " + str(((np.min(EW_returns))) * 12))
print("EW portfolio VaR : " + str(var_gaussian(EW_returns)))
print("EW portfolio ES : " + str(
    ES(EW_returns.mean(), np.sqrt(np.dot(equal_weight.T, np.dot(cov_excess, equal_weight))))))

# Value weighted portfolio base on average monthly market cap on considered period 1999-2021:

VW_weight = market_cap_nafree.mean()
VW_weight /= sum(VW_weight)
VW_returns = np.mean(VW_weight * stock, axis=1)


def print_info(prtf_name, returns, cov, weights, period=12):
    print(str(prtf_name) + " annual volatility of " + str(np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * period))
    print(str(prtf_name)+" annual average return of " + str((((np.mean(returns)))) * period))
    print(str(prtf_name)+" MAX  averaged Return : " + str(((np.max(returns))) * period))
    print(str(prtf_name)+" MIN  averaged Return : " + str(((np.min(returns))) * period))
    print(str(prtf_name)+" VaR : " + str(var_gaussian(returns)))
    print(str(prtf_name)+" ES : " + str(ES(returns.mean(), np.sqrt(np.dot(weights.T, np.dot(cov, weights))))))


print_info("value weighted", VW_returns, cov_excess, VW_weight)
"""
# ---------------------------------------------------------------------------------------------------------------------
# QUESTION 5 ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

"""
# Resample to the first 5 years / 60 months :
market_cap = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Market Cap").dropna()
market_cap_nafree = market_cap.iloc[1::, 2::]
market_cap_nafree = pd.DataFrame.transpose(market_cap_nafree)
market_cap_nafree.index = pd.to_datetime(market_cap_nafree.index)
market_cap_nafree = pd.DataFrame.resample(market_cap_nafree, "M").mean()
#market_cap_nafree = market_cap_nafree.iloc[:72, :]

# pct_change = market_cap_nafree.pct_change(axis=0)
# pct_change = pct_change.iloc[1:,:]
# pct_change_mean = np.mean(pct_change, axis=0)
stock = market_cap_nafree / market_cap_nafree.shift(1)
stock = stock.iloc[1:, :]
cov_excess = stock.cov()
pct_change_mean = np.mean(stock)

def var_gaussian(r, level=10, modified=True):
    # compute the Z score assuming it was Gaussian
    z = sp.norm.ppf(level / 100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = sp.skew(r)
        k = sp.kurtosis(r)
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )
    return -(np.mean(r) + z * np.std(r))

def ES(r, cov, alpha=1):
    CVaR_n = alpha ** -1 * sp.norm.pdf(sp.norm.ppf(alpha)) * cov - r
    return CVaR_n

# MU and COV on first 6 months
def print_info(prtf_name, returns, cov, weights, period=12):
    print(str(prtf_name) + " annual volatility of " + str(np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * period))
    print(str(prtf_name)+" annual average return of " + str((((np.mean(returns)))) * period))
    print(str(prtf_name)+" MAX  Return : " + str(((np.max(returns))) * period))
    print(str(prtf_name)+" MIN  Return : " + str(((np.min(returns))) * period))
    print(str(prtf_name)+" VaR (on period data): " + str(var_gaussian(returns)))
    print(str(prtf_name)+" ES (on period data): " + str(ES(np.mean(returns), np.sqrt(np.dot(weights.T, np.dot(cov, weights))))))

def return_min_var_alpha_POSNEG(mu, cov, gen=20000, sharesnumber=97):
    portfolio_returns = []
    portfolio_volatilities = []
    weights_vec = []
    for x in range(gen):
        weights = np.random.uniform(-1, 1, sharesnumber)
        weights /= np.sum(weights)
        portfolio_returns.append(np.sum(mu * weights))
        portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
        weights_vec.append(weights)
    portfolio_returns = np.array(portfolio_returns)
    portfolio_volatilities = np.array(portfolio_volatilities)
    weights_vec = np.array(weights_vec)
    index_min = np.argmin(portfolio_volatilities)
    ret = portfolio_returns[index_min]
    alpha = weights_vec[index_min]
    return alpha


saved_returns = []
saved_covariances = []
saved_alphas = []

for i in range(204):
    market_cap_sixyear = market_cap_nafree.iloc[i:i + 72, :]
    stock = market_cap_sixyear / market_cap_sixyear.shift(1)
    stock = stock.iloc[1:, :]
    cov_excess = stock.cov()
    pct_change_mean = np.mean(stock)
    print(str(i)+"/203 monthly periods completed")
    alpha = return_min_var_alpha_POSNEG(pct_change_mean, cov_excess)
    saved_returns.append(pct_change_mean * alpha)
    saved_covariances.append(np.sqrt(np.dot(alpha.T, np.dot(cov_excess, alpha))))
    saved_alphas.append(alpha)

#print(print_info("First 6 year out of sample GMVP",saved_returns[0],saved_covariances[0], saved_alphas[0]))
Poos_returns = []

for i in range (len(saved_returns)):
  Poos_returns.append(saved_returns[i].mean())
print(Poos_returns)

print(print_info("Poos portfolio on 6 year rolling window GMVP",Poos_returns,cov_excess, saved_alphas[np.argmin(saved_covariances)]))

saved_returns = []
saved_covariances = []
saved_alphas = []

VW_weight = market_cap_nafree.mean()
VW_weight /= sum(VW_weight)
VW_returns = np.mean(VW_weight * stock, axis=1)

for i in range(204):
    market_cap_sixyear = market_cap_nafree.iloc[i:i + 72, :]
    stock = market_cap_sixyear / market_cap_sixyear.shift(1)
    stock = stock.iloc[1:, :]
    cov_excess = stock.cov()
    pct_change_mean = np.mean(stock)
    print(str(i)+"/203 monthly periods completed")
    alpha = VW_weight
    saved_returns.append(pct_change_mean * alpha)
    saved_covariances.append(np.sqrt(np.dot(alpha.T, np.dot(cov_excess, alpha))))
    saved_alphas.append(alpha)

print(print_info("Value weighted rolling window portfolio",saved_returns, cov_excess,VW_weight))



saved_returns = []
saved_covariances = []
saved_alphas = []
equal_weight = np.full(97, 1 / 97)

for i in range(204):
    market_cap_sixyear = market_cap_nafree.iloc[i:i + 72, :]
    stock = market_cap_sixyear / market_cap_sixyear.shift(1)
    stock = stock.iloc[1:, :]
    cov_excess = stock.cov()
    pct_change_mean = np.mean(stock)
    print(str(i)+"/203 monthly periods completed")
    alpha = equal_weight
    saved_returns.append(pct_change_mean * alpha)
    saved_covariances.append(np.sqrt(np.dot(alpha.T, np.dot(cov_excess, alpha))))
    saved_alphas.append(alpha)

print(print_info("Value weighted rolling window portfolio",saved_returns, cov_excess,equal_weight))


print(saved_returns)
print(len(saved_returns))

"""

# ----------------------------------------------------------------------------------------------------------------------
# QUESTION 6------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# #the function under is written for positive weights only,
# Resample to the first 5 years / 60 months :
"""
market_cap = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Market Cap").dropna()
market_cap_nafree = market_cap.iloc[1::, 2::]
market_cap_nafree = pd.DataFrame.transpose(market_cap_nafree)
market_cap_nafree.index = pd.to_datetime(market_cap_nafree.index)
market_cap_nafree = pd.DataFrame.resample(market_cap_nafree, "M").mean()
#market_cap_nafree = market_cap_nafree.iloc[:72, :]
def sigma(mu):
    sigm = np.sqrt(np.dot(mu.T, np.dot(cov_excess, mu)))
    return sigm

# pct_change = market_cap_nafree.pct_change(axis=0)
# pct_change = pct_change.iloc[1:,:]
# pct_change_mean = np.mean(pct_change, axis=0)
stock = market_cap_nafree / market_cap_nafree.shift(1)
stock = stock.iloc[1:, :]
cov_excess = stock.cov()
pct_change_mean = np.mean(stock)

def return_min_var_alpha_POS(mu, cov, gen=30000, sharesnumber = 97):

    portfolio_returns = []
    portfolio_volatilities = []
    weights_vec = []
    for x in range(gen):
        weights = np.random.uniform(0, 1, sharesnumber)
        weights /= np.sum(weights)
        portfolio_returns.append(np.sum(mu * weights))
        portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
        weights_vec.append(weights)
    portfolio_returns = np.array(portfolio_returns)
    portfolio_volatilities = np.array(portfolio_volatilities)
    weights_vec = np.array(weights_vec)
    index_min = np.argmin(portfolio_volatilities)
    ret = portfolio_returns[index_min]
    alpha = weights_vec[index_min]
    return alpha


def var_gaussian(r, level=10, modified=True):
    # compute the Z score assuming it was Gaussian
    z = sp.norm.ppf(level / 100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = sp.skew(r)
        k = sp.kurtosis(r)
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )
    return -(np.mean(r) + z * np.std(r))

def ES(r, cov, alpha=1):
    CVaR_n = alpha ** -1 * sp.norm.pdf(sp.norm.ppf(alpha)) * cov - r
    return CVaR_n

# MU and COV on first 6 months
def print_info(prtf_name, returns, cov, weights, period=12):
    print(str(prtf_name) + " annual volatility of " + str(np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * period))
    print(str(prtf_name)+" annual average return of " + str((((np.mean(returns)))) * period))
    print(str(prtf_name)+" MAX  Return : " + str(((np.max(returns))) * period))
    print(str(prtf_name)+" MIN  Return : " + str(((np.min(returns))) * period))
    print(str(prtf_name)+" VaR (on period data): " + str(var_gaussian(returns)))
    print(str(prtf_name)+" ES (on period data): " + str(ES(np.mean(returns), np.sqrt(np.dot(weights.T, np.dot(cov, weights))))))

saved_returns = []
saved_covariances = []
saved_alphas = []
alpha = np.full(97, 1)

for i in range(204):
    market_cap_sixyear = market_cap_nafree.iloc[i:i + 72, :]
    stock = market_cap_sixyear / market_cap_sixyear.shift(1)
    stock = stock.iloc[1:, :]
    cov_excess = stock.cov()
    pct_change_mean = np.mean(stock)
    print(str(i)+"/203 monthly periods completed")
    alpha = return_min_var_alpha_POS(pct_change_mean,cov_excess)
    alpha=alpha.x
    saved_returns.append(pct_change_mean * alpha)
    saved_covariances.append(np.sqrt(np.dot(alpha.T, np.dot(cov_excess, alpha))))
    saved_alphas.append(alpha)

#print(print_info("First 6 year out of sample GMVP",saved_returns[0],saved_covariances[0], saved_alphas[0]))
Poos_returns = []

for i in range (len(saved_returns)):
  Poos_returns.append(saved_returns[i].mean())
print(Poos_returns)

print(print_info("Poos portfolio on 6 year rolling window GMVP",Poos_returns,cov_excess, saved_alphas[np.argmin(saved_covariances)]))


saved_returns = []
saved_covariances = []
saved_alphas = []

VW_weight = market_cap_nafree.mean()
VW_weight /= sum(VW_weight)
VW_returns = np.mean(VW_weight * stock, axis=1)

for i in range(204):
    market_cap_sixyear = market_cap_nafree.iloc[i:i + 72, :]
    stock = market_cap_sixyear / market_cap_sixyear.shift(1)
    stock = stock.iloc[1:, :]
    cov_excess = stock.cov()
    pct_change_mean = np.mean(stock)
    print(str(i)+"/203 monthly periods completed")
    alpha = VW_weight
    saved_returns.append(pct_change_mean * alpha)
    saved_covariances.append(np.sqrt(np.dot(alpha.T, np.dot(cov_excess, alpha))))
    saved_alphas.append(alpha)

print(print_info("Value weighted rolling window portfolio",saved_returns, cov_excess,VW_weight))



saved_returns = []
saved_covariances = []
saved_alphas = []
equal_weight = np.full(97, 1 / 97)

for i in range(204):
    market_cap_sixyear = market_cap_nafree.iloc[i:i + 72, :]
    stock = market_cap_sixyear / market_cap_sixyear.shift(1)
    stock = stock.iloc[1:, :]
    cov_excess = stock.cov()
    pct_change_mean = np.mean(stock)
    print(str(i)+"/203 monthly periods completed")
    alpha = equal_weight
    saved_returns.append(pct_change_mean * alpha)
    saved_covariances.append(np.sqrt(np.dot(alpha.T, np.dot(cov_excess, alpha))))
    saved_alphas.append(alpha)

print(print_info("Value weighted rolling window portfolio",saved_returns, cov_excess,equal_weight))



"""
# ----------------------------------------------------------------------------------------------------------------------
# QUESTION 7------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

"""
#DATA taken
co2 = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")
revenue = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Revenue")

co2 = co2.iloc[::, 2::]
co2 = pd.DataFrame.transpose(co2)
co2.index = pd.to_datetime(co2.index)
co2 = pd.DataFrame.resample(co2, "YS").mean()
co2 = co2.iloc[21,::]

revenue = revenue.iloc[::, 2::]
revenue = pd.DataFrame.transpose(revenue)
revenue.index = pd.to_datetime(revenue.index)
revenue = pd.DataFrame.resample(revenue, "YS").mean()
revenue = revenue.iloc[21,::]

c_intensity = co2/revenue
c_intensity = c_intensity.dropna()
print(c_intensity)

#portfolio with positive weights with only those with a carbon intensity surveyed


"""
#----------------------------------------------------------------------------------------------------------------------
#QUESTION 8   -------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

co2 = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")
revenue = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Revenue")
market_cap = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Market Cap").dropna()

market_cap_nafree = market_cap.iloc[1::, 2::]
market_cap_nafree = pd.DataFrame.transpose(market_cap_nafree)
market_cap_nafree.index = pd.to_datetime(market_cap_nafree.index)
market_cap_nafree = pd.DataFrame.resample(market_cap_nafree, "M").mean()
market_cap_nafree = market_cap_nafree.T

co2 = co2.iloc[::, 2::]
co2 = pd.DataFrame.transpose(co2)
co2.index = pd.to_datetime(co2.index)
co2 = pd.DataFrame.resample(co2, "YS").mean()
co2 = co2.iloc[21,::]

revenue = revenue.iloc[::, 2::]
revenue = pd.DataFrame.transpose(revenue)
revenue.index = pd.to_datetime(revenue.index)
revenue = pd.DataFrame.resample(revenue, "YS").mean()
revenue = revenue.iloc[21,::]

c_intensity = co2/revenue
c_intensity = c_intensity.dropna()
c_intensity = c_intensity.T
# Merging carbon intensity with the residual informations about revenue on the same period
print(c_intensity.T)
print(market_cap_nafree)
merged_co2_cap = market_cap_nafree.merge(c_intensity, left_index = True, right_index = True)
print(merged_co2_cap)

Portfolio_value = 1000000

#Reusing the initial minimal variance portfolio with positive weights













































































































































