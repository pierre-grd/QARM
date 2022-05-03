import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = "new number"

# co_2 = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")

market_cap = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Market Cap")
# y = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")
feuille1 = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Feuille 1 - Group_P")
revenue = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Revenue")
sic = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="SIC")

#market_cap = market_cap.merge(sic)
#print(market_cap)
# tt_return_index = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="TT Return Index")
#for i in range(1, 149):
  #print(revenue.iloc[i, 1:277].describe())

#150*278

print(feuille1)

#market_cap = market_cap.merge(feuille1)

# print(y.iloc[::1])

# Question 2 - KC TRY --------------------

#revenue = revenue.dropna()
#print(market_cap)






# Question 3 -----------------------



