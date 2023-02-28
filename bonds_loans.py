import pandas as pd

bonds_loans = pd.read_parquet('debt.parquet')

# sort the data, getting only the values that are the most recent, not all duplicates as well
bonds_loans = bonds_loans[bonds_loans["latestfilingforinstanceflag"].isin([1])]
bonds_loans = bonds_loans[bonds_loans["latestforfinancialperiodflag"].isin([1])]

# make the "periodenddate" into date format
bonds_loans['periodenddate'] = bonds_loans['periodenddate'].apply(lambda x: pd.to_datetime(str(int(x))))

# creating the "quarter" column based on "periodenddate" displaying 1-4
bonds_loans['quarter'] = bonds_loans['periodenddate'].dt.to_period('Q').dt.quarter

# Created new dataframe for sum of quarterly debt payments
summaryq = bonds_loans.groupby('quarter')['dataitemvalue'].sum().reset_index()
quarter_list = bonds_loans['quarter'].tolist()
#quarter_list = bonds_loans.loc[bonds_loans['quarter'] == 4]
# adding more columns to summaryq
quarter4 = quarter_list == "4"

newtest = quarter4.groupby('companyid')['dataitemvalue'].sum().reset_index()
#summaryq['annualdebtpaid'] = quarter4.groupby('companyid')['dataitemvalue'].sum().reset_index()