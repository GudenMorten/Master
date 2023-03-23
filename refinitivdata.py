import pandas as pd
# import pylab as pl
import seaborn as sns
import matplotlib as plt
from polars.polars import cols

refinitivdata = pd.read_csv('refinitivdata2.csv')

# fixing the dataframe
refinitivdata[['Company Common Name', 'NAICS Sector Code', 'NAICS Sector Name',
               'NAICS Subsector Code', 'NAICS Subsector Name', 'NAICS National Industry Code',
               'NAICS National Industry Name', 'NAICS Industry Group Code', 'NAICS Industry Group Name',
               'Country of Exchange', 'Exchange Name', 'Market Capitalization', 'Net Income after Tax',
               "Total Shareholders' Equity incl Minority Intr & Hybrid Debt"]] = refinitivdata.groupby('Instrument')[
    ['Company Common Name', 'NAICS Sector Code',
     'NAICS Sector Name', 'NAICS Subsector Code', 'NAICS Subsector Name', 'NAICS National Industry Code',
     'NAICS National Industry Name', 'NAICS Industry Group Code', 'NAICS Industry Group Name',
     'Country of Exchange', 'Exchange Name', 'Market Capitalization', 'Net Income after Tax',
     "Total Shareholders' Equity incl Minority Intr & Hybrid Debt"]].fillna(method='ffill')

# make the Date column accessible with date format
refinitivdata['Date'] = pd.to_datetime(refinitivdata['Date'], format='%Y/%m/%d')

# exclude rows with value 22, 52, 53 in the "NAICS Sector Code" column
refinitivdata = refinitivdata[~refinitivdata["NAICS Sector Code"].isin(['22', '52', '53'])]
refinitivdata = refinitivdata[~refinitivdata["Date"].isin(["NaT"])]
refinitivdata = refinitivdata[refinitivdata["Country of Exchange"].isin(['Norway', 'Sweden', 'Finland', 'Denmark'])]

# Create new column for long-term debt/tot-debt
refinitivdata['Long-Term Debt/Total Debt'] = refinitivdata['Debt - Long-Term - Total'] / refinitivdata['Debt - Total']
# Create short term debt and current long-term debt over total debt column
refinitivdata['Short-Term Debt/Total Debt'] = refinitivdata['Short-Term Debt & Current Portion of Long-Term Debt'] / \
                                              refinitivdata['Debt - Total']
# Create a 'YEAR' column for easier use of plots
refinitivdata['Fiscal Year'] = refinitivdata['Date'].dt.year
refinitivdata['ROE'] = refinitivdata['Net Income after Tax'] / refinitivdata[
    "Total Shareholders' Equity incl Minority Intr & Hybrid Debt"]

# Merging the gvisin dataset with the refinitivdata dataset to include gvkey as well as ISIN number
refinitivdata_withgvkey = pd.merge(refinitivdata, gvisin, on='Instrument', how='left')

# merging refinitivdata_withgvkey with capitalstructure_sorted
combined_dataset = pd.merge(refinitivdata_withgvkey, capitalstructure_sorted, left_on=['gvkey', 'Fiscal Year'],
                            right_on=['gvkey', 'year'], how='right')
combined_dataset = combined_dataset[~combined_dataset["NAICS Sector Code"].isin(['22', '52', '53'])]
combined_dataset = combined_dataset[~combined_dataset["Date"].isin(["NaT"])]
combined_dataset = combined_dataset[
    combined_dataset["Country of Exchange"].isin(['Norway', 'Sweden', 'Finland', 'Denmark'])]

combined_dataset.fillna(0, inplace=True)
# renaming columns for combined_dataset
combined_dataset.rename(columns={'1': 'Commercial Paper'}, inplace=True)
combined_dataset.rename(columns={'2': 'Revolving Credit'}, inplace=True)
combined_dataset.rename(columns={'3': 'Term Loans'}, inplace=True)
combined_dataset.rename(columns={'4': 'Bonds and Notes'}, inplace=True)
combined_dataset.rename(columns={'5': 'Capital Lease'}, inplace=True)
combined_dataset.rename(columns={'6': 'Trust Preferred'}, inplace=True)
combined_dataset.rename(columns={'7': 'Other Borrowings'}, inplace=True)

combined_dataset['Commercial Paper'] = combined_dataset['Commercial Paper'].apply(lambda x: int(x))
combined_dataset['Revolving Credit'] = combined_dataset['Revolving Credit'].apply(lambda x: int(x))
combined_dataset['Term Loans'] = combined_dataset['Term Loans'].apply(lambda x: int(x))
combined_dataset['Bonds and Notes'] = combined_dataset['Bonds and Notes'].apply(lambda x: int(x))
combined_dataset['Capital Lease'] = combined_dataset['Capital Lease'].apply(lambda x: int(x))
combined_dataset['Trust Preferred'] = combined_dataset['Trust Preferred'].apply(lambda x: int(x))
combined_dataset['Other Borrowings'] = combined_dataset['Other Borrowings'].apply(lambda x: int(x))

combined_dataset['Total Debt CapitalIQ'] = combined_dataset[
    ['Revolving Credit', 'Term Loans', 'Bonds and Notes', 'Commercial Paper', 'Capital Lease', 'Other Borrowings',
     'Trust Preferred']].fillna(0).sum(axis=1)

combined_dataset['Revolving Credit/Total Debt'] = combined_dataset['Revolving Credit'] / combined_dataset[
    'Total Debt CapitalIQ']
combined_dataset['Term Loans/Total Debt'] = combined_dataset['Term Loans'] / combined_dataset['Total Debt CapitalIQ']
combined_dataset['Bonds and Notes/Total Debt'] = combined_dataset['Bonds and Notes'] / combined_dataset[
    'Total Debt CapitalIQ']
combined_dataset['Commercial Paper/Total Debt'] = combined_dataset['Commercial Paper'] / combined_dataset[
    'Total Debt CapitalIQ']
combined_dataset['Capital Lease/Total Debt'] = combined_dataset['Capital Lease'] / combined_dataset[
    'Total Debt CapitalIQ']
combined_dataset['Other Borrowings/Total Debt'] = combined_dataset['Other Borrowings'] / combined_dataset[
    'Total Debt CapitalIQ']
combined_dataset['Trust Preferred/Total Debt'] = combined_dataset['Trust Preferred'] / combined_dataset[
    'Total Debt CapitalIQ']

combined_dataset['Total debt relative'] = combined_dataset[
    ['Revolving Credit/Total Debt', 'Term Loans/Total Debt', 'Bonds and Notes/Total Debt',
     'Commercial Paper/Total Debt', 'Capital Lease/Total Debt', 'Other Borrowings/Total Debt',
     'Trust Preferred/Total Debt']].fillna(0).sum(axis=1)


debt_specialization = combined_dataset[
    ['year', 'Revolving Credit/Total Debt', 'Term Loans/Total Debt', 'Bonds and Notes/Total Debt',
     'Commercial Paper/Total Debt', 'Capital Lease/Total Debt', 'Other Borrowings/Total Debt',
     'Trust Preferred/Total Debt']]
debt_specialization_polar = pl.from_pandas(
    debt_specialization[
        ['year', 'Revolving Credit/Total Debt', 'Term Loans/Total Debt', 'Bonds and Notes/Total Debt',
         'Commercial Paper/Total Debt', 'Capital Lease/Total Debt', 'Other Borrowings/Total Debt',
         'Trust Preferred/Total Debt']])

debt_specialization_polar = debt_specialization_polar.groupby(
    [
        "year"
    ]
).agg(
    [
        pl.avg("Term Loans/Total Debt").alias("Term Loans"),
        pl.avg("Bonds and Notes/Total Debt").alias("Bonds and Notes"),
        pl.avg("Revolving Credit/Total Debt").alias("Revolving Credit"),
        pl.avg("Other Borrowings/Total Debt").alias("Other Borrowings"),
        pl.avg("Capital Lease/Total Debt").alias("Capital Lease"),
        pl.avg("Commercial Paper/Total Debt").alias("Commercial Paper"),
        pl.avg("Trust Preferred/Total Debt").alias("Trust Preferred"),
    ]
).to_pandas().fillna(0)

debt_specialization_polar['Total'] = debt_specialization_polar["Term Loans"] + debt_specialization_polar[
    "Bonds and Notes"] + debt_specialization_polar["Revolving Credit"] + debt_specialization_polar["Other Borrowings"] + \
                                     debt_specialization_polar["Capital Lease"] + debt_specialization_polar[
                                         "Commercial Paper"] + debt_specialization_polar["Trust Preferred"]
debt_specialization_polar = debt_specialization_polar.sort_values(by='year', ascending=True)
debt_specialization_polar = debt_specialization_polar.set_index('year')
debt_specialization_polar = debt_specialization_polar.transpose()

combined_dataset['Unique debts'] = combined_dataset[
    ["Term Loans", "Bonds and Notes", "Revolving Credit", "Other Borrowings", "Capital Lease", "Commercial Paper",
     "Trust Preferred"]].apply(lambda x: sum(x != 0), axis=1)


########### Herfindahl Index #############
def HHI(df):
    RC = df['Revolving Credit']
    TL = df['Term Loans']
    BN = df['Bonds and Notes']
    CL = df['Capital Lease']
    CP = df['Commercial Paper']
    TP = df['Trust Preferred']
    OTHER = df['Other Borrowings']
    df['TD'] = df[[RC, TL, BN, CL, CP, TP, OTHER]].sum(axis=1)

    df['SS'] = (df[RC] / df['TD']) ** 2 + (df[TL] / df['TD']) ** 2 + \
               (df[BN] / df['TD']) ** 2 + (df[CL] / df['TD']) ** 2 + \
               (df[CP] / df['TD']) ** 2 + (df[TP] / df['TD']) ** 2 + \
               (df[OTHER] / df['TD']) ** 2

    # Calculate HHI
    df['HHI'] = (df['SS'] - (1 / 7)) / (1 - (1 / 7))

HHI(combined_dataset)