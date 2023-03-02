import pandas as pd
import seaborn as sns
import matplotlib as plt

refinitivdata = pd.read_csv('refinitivdata.csv')

# fixing the dataframe
refinitivdata[['Company Common Name', 'NAICS Sector Code', 'NAICS Sector Name',
    'NAICS Subsector Code', 'NAICS Subsector Name', 'NAICS National Industry Code',
    'NAICS National Industry Name', 'NAICS Industry Group Code', 'NAICS Industry Group Name',
    'Country of Exchange', 'Exchange Name']] = refinitivdata.groupby('Instrument')[['Company Common Name', 'NAICS Sector Code',
    'NAICS Sector Name', 'NAICS Subsector Code', 'NAICS Subsector Name', 'NAICS National Industry Code',
    'NAICS National Industry Name', 'NAICS Industry Group Code', 'NAICS Industry Group Name',
    'Country of Exchange', 'Exchange Name']].fillna(method='ffill')

# make the Date column accessible with date format
refinitivdata['Date'] = pd.to_datetime(refinitivdata['Date'], format='%Y/%m/%d')

# exclude rows with value 22, 52, 53 in the "NAICS Sector Code" column
refinitivdata = refinitivdata[~refinitivdata["NAICS Sector Code"].isin(['22', '52', '53'])]
refinitivdata = refinitivdata[~refinitivdata["Date"].isin(["NaT"])]
refinitivdata = refinitivdata[refinitivdata["Country of Exchange"].isin(['Norway', 'Sweden', 'Finland', 'Denmark'])]


# Create new column for long-term debt/tot-debt
refinitivdata['Long-Term Debt/Total Debt'] = refinitivdata['Debt - Long-Term - Total'] / refinitivdata['Debt - Total']
# Create short term debt and current long-term debt over total debt column
refinitivdata['Short-Term Debt/Total Debt'] = refinitivdata['Short-Term Debt & Current Portion of Long-Term Debt'] / refinitivdata['Debt - Total']
# Create a 'YEAR' column for easier use of plots
refinitivdata['Fiscal Year'] = refinitivdata['Date'].dt.year


chart = sns.barplot(x='Fiscal Year', y='Long-Term Debt/Total Debt', hue= 'Country of Exchange', data=refinitivdata)
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
plt.pyplot.show()