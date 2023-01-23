import eikon as ek
import pandas as pd

# Define the Eikon App Key
ek.set_app_key("62297b98ae4947f2ac01401801b0c32a165a1053")

# Define the debt structure financials fields
fields = ['TR.DebtStructure.Financials.IssueSize', 'TR.DebtStructure.Financials.IssueCurrency', 'TR.DebtStructure.Financials.IssueType', 'TR.DebtStructure.Financials.Maturity']

# Define the Nordic region
region = '.NORDIC'

# Define the RICs of the companies you want to retrieve data for
rics = ['ERICB.ST','TEL.OL','NDA.ST']

# Retrieve the data
data = ek.get_data(rics, fields, {'Region': region})

# Print the data
print(data)

# You can also save the data to a csv file
# data.to_csv("DebtStructureFinancialsData.csv")