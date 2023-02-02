import eikon as ek
import pandas as pd

# Replace YOUR_APP_KEY with your Eikon API key
ek.set_app_key('022971ea1eb44687b0bcb92bad73c9b83ab08a9d')

# Define the ISIN identifier for the company
isin = 'NO0005052605'

# Use the Eikon Data API to retrieve the company data
# data = ek.get_data(isin, ['TR.Revenue', 'TR.NetIncomeLoss'])
df, err = ek.get_data(isin, fields=['TR.PriceClose', 'TR.F.DebtTot'],
                   parameters={'SDate': '1990-01-01', 'EDate': '2022-12-31', "Frq": "Y"}, debug=True)


# Print the DataFrame
print(df.head(20))

nordic_data = pd.read_excel("nordic_data_bonds.xlsx")
list1 = nordic_data["Preferred RIC"].dropna().to_list()
df2, _ =  ek.get_data(list1, fields=['TR.PriceClose', 'TR.F.DebtTot'],
                   parameters={'SDate': '1990-01-01', 'EDate': '2022-12-31', "Frq": "Y"})