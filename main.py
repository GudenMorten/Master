import pandas as pd
import seaborn as sns
import matplotlib as plt

dataf2 = pd.read_csv('gathered_data2.csv')

# fixing the dataframe
dataf2[['Company Common Name', 'NAICS Sector Code', 'NAICS Sector Name',
    'NAICS Subsector Code', 'NAICS Subsector Name', 'NAICS National Industry Code',
    'NAICS National Industry Name', 'NAICS Industry Group Code', 'NAICS Industry Group Name',
    'Country of Exchange', 'Exchange Name']] = dataf2.groupby('Instrument')[['Company Common Name', 'NAICS Sector Code',
    'NAICS Sector Name', 'NAICS Subsector Code', 'NAICS Subsector Name', 'NAICS National Industry Code',
    'NAICS National Industry Name', 'NAICS Industry Group Code', 'NAICS Industry Group Name',
    'Country of Exchange', 'Exchange Name']].fillna(method='ffill')

# make the Date column accessible with date format
dataf2['Date'] = pd.to_datetime(dataf2['Date'], format='%Y/%m/%d')

# exclude rows with value 22, 52, 53 in the "NAICS Sector Code" column
dataf2 = dataf2[~dataf2["NAICS Sector Code"].isin(['22', '52', '53'])]
dataf2 = dataf2[~dataf2["Date"].isin(["NaT"])]


# Create new column for long-term debt/tot-debt
dataf2['Long-Term Debt/Total Debt'] = dataf2['Debt - Long-Term - Total']/dataf2['Debt - Total']
# Create short term debt and current long-term debt over total debt column
dataf2['Short-Term Debt/Total Debt'] = dataf2['Short-Term Debt & Current Portion of Long-Term Debt']/dataf2['Debt - Total']
# Create a 'YEAR' column for easier use of plots
dataf2['Fiscal Year'] = dataf2['Date'].dt.year
#print(dataf2)

#sns.lmplot(dataf2['Total Debt Percentage of Total Equity'].groupby(dataf2['Country of Exchange']))
chart = sns.barplot(x='Date', y='Long-Term Debt/Total Debt', hue= 'Country of Exchange', data=dataf2)
plt.pyplot.show()