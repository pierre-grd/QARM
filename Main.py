
import os
import pandas as pd
x = "new number"

y = pd.read_excel(os.path.join("C:/Users/pierr/Documents/GitHub/QARM", "Data QARM.xlsx"), engine="openpyxl")
print(y)


