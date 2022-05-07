import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


# Question 1 - Seperate Data by Sector : Extrapolate 3 most represented and Analyze Mean, Variance,
# skewness, kurtosis, minimum, and maximum.
#-------------------------------------------

#Creat List of GIC sectors so as to find top 3

mylist = feuille1['GICSSector'].tolist()
#print(mylist)

import collections
c = collections.Counter(mylist)
print(c.most_common(3))

#We now know top 3 sectors are Industrials, Financials and Consumer Discretionary

#Delete every company that is not part of the 3 sectors

#Industrials = market_cap()








# Question 2 - KC TRY --------------------
market_cap = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Market Cap").dropna()
market_cap_nafree = market_cap.iloc[1::,2::]

#DATA CLEANING & Montly scaled :

market_cap_nafree = pd.DataFrame.transpose(market_cap_nafree)
market_cap_nafree.index = pd.to_datetime(market_cap_nafree.index)
market_cap_nafree = pd.DataFrame.resample(market_cap_nafree, "M").mean()
pct_change = market_cap_nafree.pct_change(axis=0)
pct_change = pct_change.iloc[1:,:]
pct_change_mean = np.mean(pct_change, axis=0)


stock = market_cap_nafree/market_cap_nafree.shift(1)
stock = stock.iloc[1:,:]
stock = stock
print(stock)
cov_excess = stock.cov()
pct_change_mean = np.mean(stock)


#Create a list of randomized weighting vectors :
portfolio_returns = []
portfolio_volatilities = []

#replace 97 by the exact number of companies's stock in the portfolio
for x in range(100000):
    weights = np.random.random(97)
    weights /= np.sum(weights)
    portfolio_returns.append(np.sum(pct_change_mean*weights))
    portfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(cov_excess,weights))))

portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)

print(portfolio_returns)
print(portfolio_volatilities)

portfolios_frt = pd.DataFrame({"Return" : portfolio_returns,"Volatility":portfolio_volatilities})

portfolios_frt.plot(x="Volatility", y="Return", kind="scatter", color="blue", s=4)
plt.xlabel("Expected Volatility")
plt.ylabel("Expected Return")
plt.show()







# Question 3 -----------------------


#revenue = revenue.dropna()
#print(revenue)




