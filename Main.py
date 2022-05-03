import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


co_2 = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")
market_cap = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Market Cap")
feuille1 = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Feuille 1 - Group_P")
revenue = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Revenue")
sic = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="SIC")
tt_return_index = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="TT Return Index")

#market_cap = market_cap.merge(sic)
#print(market_cap)
#market_cap = market_cap.merge(sic, how="left")

market_cap = market_cap.merge(feuille1, how="left", on='ISIN')
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

Industrials = market_cap()








# Question 2 - KC TRY --------------------

#market_cap = market_cap.dropna()
#print(market_cap)






# Question 3 -----------------------



