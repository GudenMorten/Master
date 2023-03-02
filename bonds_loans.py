import pandas as pd
import seaborn as sns
import matplotlib as plt


bonds_loans = pd.read_parquet('debt.parquet')

# sort the data, getting only the values that are the most recent, not all duplicates as well
bonds_loans = bonds_loans[bonds_loans["latestfilingforinstanceflag"].isin([1])]
bonds_loans = bonds_loans[bonds_loans["latestforfinancialperiodflag"].isin([1])]

# make the "periodenddate" and 'filingdate' into date format
bonds_loans['periodenddate'] = bonds_loans['periodenddate'].apply(lambda x: pd.to_datetime(str(int(x))))
bonds_loans['filingdate'] = bonds_loans['filingdate'].apply(lambda x: pd.to_datetime(str(int(x))))

# creating and fixing certain columns
bonds_loans['quarter'] = bonds_loans['periodenddate'].dt.to_period('Q')
bonds_loans['quarternumb'] = bonds_loans['periodenddate'].dt.to_period('Q').dt.quarter
bonds_loans['year'] = bonds_loans['periodenddate'].dt.year
bonds_loans['gvkey'] = bonds_loans['gvkey'].apply(lambda x: (int(x)))


# Created new dataframe for sum of quarterly debt payments
#summaryq = bonds_loans.groupby('quarter')['dataitemvalue'].sum().reset_index()
#summaryq = pd.DataFrame()
#summaryq['quarter'] = bonds_loans['quarter']
#summaryq["year"] = bonds_loans['year']
#summaryq["dataitemvalue"] = bonds_loans['dataitemvalue']
#summaryq["capitalstructuresubtypeid"] = bonds_loans['capitalstructuresubtypeid']
#quarter_list = bonds_loans['quarter'].tolist()
#quarter_list = bonds_loans.loc[bonds_loans['quarter'] == 4]
# adding more columns to summaryq
#quarter4 = quarter_list == "4"

#newtest = quarter4.groupby('companyid')['dataitemvalue'].sum().reset_index()
#summaryq['annualdebtpaid'] = quarter4.groupby('companyid')['dataitemvalue'].sum().reset_index()

########################
# PLOTTING
#chart = sns.barplot(x='periodenddate', y='dataitemvalue', hue= 'capitalstructuredescription', data=bonds_loans)
#plt.pyplot.show()

#chart = sns.barplot(x='year', y='dataitemvalue', hue='capitalstructuresubtypeid', data=summaryq)
#chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
#plt.pyplot.show()