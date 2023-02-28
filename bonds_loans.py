import pandas as pd

bonds_loans = pd.read_parquet('debt.parquet')

# sort the data
bonds_loans = bonds_loans[bonds_loans["latestfilingforinstanceflag"].isin([1])]
bonds_loans = bonds_loans[bonds_loans["latestforfinancialperiodflag"].isin([1])]

# make the Date column accessible with date format
bonds_loans['Date'] = pd.to_datetime(bonds_loans['periodenddate'], format='%Y/%m/%d')

#summaryq =