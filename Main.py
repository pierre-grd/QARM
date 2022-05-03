import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = "new number"

# co_2 = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")

# market_cap = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="Market Cap")
# y = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")

revenue = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="Revenue")

# tt_return_index = pd.read_excel("Data QARM.xlsx", engine="openpyxl", sheet_name="TT Return Index")
for i in range(1, 149):
  print(revenue.iloc[i, 1:277].describe())

#150*278

print(len(revenue.columns))


<<<<<<< HEAD

print(y.iloc[::1])

# Question 2



=======
>>>>>>> 0731837c0533c7a4f32e51c9bb894a76262d7de1
