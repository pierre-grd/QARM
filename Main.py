import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import scipy.stats as sp



"""
#co_2 = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")
feuille1 = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Feuille 1 - Group_P")
revenue = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Revenue")
sic = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="SIC")
tt_return_index = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="TT Return Index")

#market_cap = market_cap.merge(sic)
#print(market_cap)
#market_cap = market_cap.merge(sic, how="left")

#market_cap = market_cap.merge(feuille1, how="left", on='ISIN')
#print(market_cap)

#----------------------------------------------------------------------------------------------------------------------
# Question 1 - --------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------

Seperate Data by Sector : Extrapolate 3 most represented and Analyze Mean, Variance,
# skewness, kurtosis, minimum, and maximum.

#Creat List of GIC sectors so as to find top 3

mylist = feuille1['GICSSector'].tolist()
#print(mylist)

import collections
c = collections.Counter(mylist)
print(c.most_common(3))

#We now know top 3 sectors are Industrials, Financials and Consumer Discretionary

#Delete every company that is not part of the 3 sectors

#Industrials = market_cap()

"""
#-----------------------------------------------------------------------------------------------------------------------
# Question 2 -----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

market_cap = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Market Cap").dropna()
market_cap_nafree = market_cap.iloc[1::,2::]

#DATA CLEANING & Montly scaled :

market_cap_nafree = pd.DataFrame.transpose(market_cap_nafree)
market_cap_nafree.index = pd.to_datetime(market_cap_nafree.index)
market_cap_nafree = pd.DataFrame.resample(market_cap_nafree, "M").mean()
#pct_change = market_cap_nafree.pct_change(axis=0)
#pct_change = pct_change.iloc[1:,:]
#pct_change_mean = np.mean(pct_change, axis=0)
print(market_cap_nafree)

stock = market_cap_nafree/market_cap_nafree.shift(1)
stock = stock.iloc[1:,:]
cov_excess = stock.cov()
pct_change_mean = np.mean(stock)
print(stock)
print(pct_change_mean)
#Create a list of randomized weighting vectors :

portfolio_returns = []
portfolio_volatilities = []
weights_vec = []
#replace 97 by the exact number of companies's stock in the portfolio
"""
def MVPs(lambd, stocks, cov):
    cov_in = np.linalg.inv(cov)
    stock_mu = stocks.shape[0]
    e = np.ones((stock_mu, 1))
    A = (cov_in @ e)/(e.T @ cov_in @ e)
    B = (1/lambd) * cov_in
    C = ((e.T @ cov_in @ stocks)/(e.T @ cov_in @ e))*e
    C = pd.DataFrame(C).squeeze()
    D = stocks - C
    alpha = A + B@C
    return alpha

for i in range (2,700):
    weights = MVPs(i/100, pct_change_mean,cov_excess)
    weights /= np.sum(weights)
    weights = weights[0]
    retur_n = weights*pct_change_mean
    volat = np.sqrt(weights.T @cov_excess@weights)
    portfolio_returns.append(retur_n)
    portfolio_volatilities.append(volat)

"""
for x in range(10000):
    weights = np.random.random(97)
    weights /= np.sum(weights)
    portfolio_returns.append(np.sum(pct_change_mean*weights))
    portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(cov_excess,weights))))
    weights_vec.append(weights)

portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)
weights_vec = np.array(weights_vec)

portfolios_frt = pd.DataFrame({"Return" : portfolio_returns,"Volatility":portfolio_volatilities})
portfolios_frt.plot(x="Volatility", y="Return", kind="scatter", color="blue", s=4)
plt.xlabel("Monthly Expected Volatility")
plt.ylabel("Monthly Expected Return")
plt.show()

"""
#-----------------------------------------------------------------------------------------------------------------------
# Question 3 -----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

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
#-----------------------------------------------------------------------------------------------------------------------
# Question 4 -----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
"""
min = np.min(portfolio_volatilities)
index_min = np.argmin(portfolio_volatilities)


print("The GMVP has an annualized volatility of " +str(min*12))
print("GMVP Annualized average return is : "+str((np.mean(portfolio_returns[index_min])-1)*12))

stock_for_mv = weights_vec[index_min] * stock

stock_for_mv = pd.DataFrame.mean(stock_for_mv, axis=1)
print("GMV portfolio MIN ANN Return: "+str(pd.DataFrame.min(stock_for_mv)*12))
print("GMV portfolio MAX ANN Return : "+str(pd.DataFrame.max(stock_for_mv)*12))


def var_gaussian(r, level=10, modified=True):
    # compute the Z score assuming it was Gaussian
    z = sp.norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = sp.skew(r)
        k = sp.kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*np.std(r))

def ES(r,cov, alpha = 1):
    CVaR_n = alpha ** -1 * sp.norm.pdf(sp.norm.ppf(alpha)) *cov  - r
    return CVaR_n

print("MVP portfolio VaR "+str(var_gaussian(stock_for_mv)))
print("MVP portfolio ES "+str(ES(stock_for_mv.mean(),
                                                 cov=portfolio_volatilities[index_min])))


equal_weight = np.full(97, 1/97)
EW_returns = np.mean(equal_weight*stock, axis=1)
print("The equally weighted portfolio have an annual volatility of "+
      str(np.sqrt(np.dot(equal_weight.T, np.dot(cov_excess,equal_weight)))*12))
print("EW portfolio average annual return : "+str((((np.mean(EW_returns))))*12))
print("EW portfolio MAX ANN Return : "+str(((np.max(EW_returns)))*12))
print("EW portfolio MIN ANN Return : "+str(((np.min(EW_returns)))*12))
print("EW portfolio VaR : "+str(var_gaussian(EW_returns)))
print("EW portfolio ES : "+str(ES(EW_returns.mean(),np.sqrt(np.dot(equal_weight.T, np.dot(cov_excess,equal_weight))))))


#Value weighted portfolio base on average monthly market cap on considered period 1999-2021:

VW_weight = market_cap_nafree.mean()
VW_weight /= sum(VW_weight)
VW_returns = np.mean(VW_weight * stock, axis=1)


print("The value weighted portfolio have an annual volatility of "+
      str(np.sqrt(np.dot(VW_weight.T, np.dot(cov_excess,VW_weight)))*12))
print("VW portfolio annual average return of "+str((((np.mean(VW_returns))))*12))
print("VW portfolio MAX ANN Return : "+str(((np.max(VW_returns)))*12))
print("VW portfolio MIN ANN Return : "+str(((np.min(VW_returns)))*12))
print("VW portfolio VaR : "+str(var_gaussian(VW_returns)))
print("VW portfolio ES : "+str(ES(VW_returns.mean(),np.sqrt(np.dot(VW_weight.T, np.dot(cov_excess,VW_weight))))))
"""
#---------------------------------------------------------------------------------------------------------------------
# QUESTION 5 ---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

