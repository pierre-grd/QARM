import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import scipy.stats as sp


co_2 = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="CO2 Emissions")
market_cap = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Market Cap").dropna()
feuille1 = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Feuille 1 - Group_P")
revenue = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="Revenue")
sic = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="SIC")
tt_return_index = pd.read_excel("Data QARM-2.xlsx", engine="openpyxl", sheet_name="TT Return Index")

print(co_2)