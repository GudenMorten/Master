import pandas as pd

gvkey = pd.read_excel("gvkey.xlsx")

gvkeylist = gvkey["gvkey"].to_list()

data4 = pd.read_parquet('capiq_Debt.parquet', engine='pyarrow')

debt = data4[data4["gvkey"].isin(gvkeylist)]

debt.to_parquet("/Users/kristin/PycharmProjects/Master/debt.parquet")
