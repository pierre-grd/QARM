import os
import pandas as pd
x = "new number"

co_2 = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")

market_cap = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="Market Cap")
y = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")

revenue = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="Revenue")

tt_return_index = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="TT Return Index")



print(y.iloc[::1])

# Question 2



