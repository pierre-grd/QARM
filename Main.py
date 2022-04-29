import os
import pandas as pd
x = "new number"

path = os.path.join("C:/Users/pierr/Documents/GitHub/QARM", "Data QARM.xlsx")

y = pd.read_excel(path, engine="openpyxl", sheet_name="CO2 Emissions")

print(y)


