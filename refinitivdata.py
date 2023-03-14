import pandas as pd
import seaborn as sns
import matplotlib as plt

refinitivdata = pd.read_csv('refinitivdata2.csv')

# fixing the dataframe
refinitivdata[['Company Common Name', 'NAICS Sector Code', 'NAICS Sector Name',
    'NAICS Subsector Code', 'NAICS Subsector Name', 'NAICS National Industry Code',
    'NAICS National Industry Name', 'NAICS Industry Group Code', 'NAICS Industry Group Name',
    'Country of Exchange', 'Exchange Name', 'Market Capitalization', 'Net Income after Tax', "Total Shareholders' Equity incl Minority Intr & Hybrid Debt"]] = refinitivdata.groupby('Instrument')[['Company Common Name', 'NAICS Sector Code',
    'NAICS Sector Name', 'NAICS Subsector Code', 'NAICS Subsector Name', 'NAICS National Industry Code',
    'NAICS National Industry Name', 'NAICS Industry Group Code', 'NAICS Industry Group Name',
    'Country of Exchange', 'Exchange Name', 'Market Capitalization', 'Net Income after Tax', "Total Shareholders' Equity incl Minority Intr & Hybrid Debt"]].fillna(method='ffill')

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
refinitivdata['ROE'] = refinitivdata['Net Income after Tax']/refinitivdata["Total Shareholders' Equity incl Minority Intr & Hybrid Debt"]

# Merging the gvisin dataset with the refinitivdata dataset to include gvkey as well as ISIN number
refinitivdata_withgvkey= pd.merge(refinitivdata, gvisin, on='Instrument', how='left')



# merging refinitivdata_withgvkey with capitalstructure_sorted
combined_dataset = pd.merge(refinitivdata_withgvkey, capitalstructure_sorted, left_on=['gvkey','Fiscal Year'], right_on=['gvkey', 'year'], how='right')
combined_dataset = combined_dataset[~combined_dataset["NAICS Sector Code"].isin(['22', '52', '53'])]
combined_dataset = combined_dataset[~combined_dataset["Date"].isin(["NaT"])]
combined_dataset = combined_dataset[combined_dataset["Country of Exchange"].isin(['Norway', 'Sweden', 'Finland', 'Denmark'])]

# renaming columns for combined_dataset
combined_dataset.rename(columns={'1': 'Commercial Paper'}, inplace=True)
combined_dataset.rename(columns={'2': 'Revolving Credit'}, inplace=True)
combined_dataset.rename(columns={'3': 'Term Loans'}, inplace=True)
combined_dataset.rename(columns={'4': 'Bonds and Notes'}, inplace=True)
combined_dataset.rename(columns={'5': 'Capital Lease'}, inplace=True)
combined_dataset.rename(columns={'6': 'Trust Preferred'}, inplace=True)
combined_dataset.rename(columns={'7': 'Other Borrowings'}, inplace=True)

combined_dataset['Total Debt CapitalIQ'] = combined_dataset[['Revolving Credit', 'Term Loans', 'Bonds and Notes', 'Commercial Paper', 'Capital Lease', 'Other Borrowings', 'Trust Preferred']].fillna(0).sum(axis=1)

combined_dataset['Revolving Credit/Total Debt'] = combined_dataset['Revolving Credit']/combined_dataset['Total Debt CapitalIQ']
combined_dataset['Term Loans/Total Debt'] = combined_dataset['Term Loans']/combined_dataset['Total Debt CapitalIQ']
combined_dataset['Bonds and Notes/Total Debt'] = combined_dataset['Bonds and Notes']/combined_dataset['Total Debt CapitalIQ']
combined_dataset['Commercial Paper/Total Debt'] = combined_dataset['Commercial Paper']/combined_dataset['Total Debt CapitalIQ']
combined_dataset['Capital Lease/Total Debt'] = combined_dataset['Capital Lease']/combined_dataset['Total Debt CapitalIQ']
combined_dataset['Other Borrowings/Total Debt'] = combined_dataset['Other Borrowings']/combined_dataset['Total Debt CapitalIQ']
combined_dataset['Trust Preferred/Total Debt'] = combined_dataset['Trust Preferred']/combined_dataset['Total Debt CapitalIQ']
