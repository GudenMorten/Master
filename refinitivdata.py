import pandas as pd
import polars as pl
import matplotlib as plt
import matplotlib.pyplot as p
from matplotlib import cm
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm
from scipy.stats.mstats import winsorize
import statsmodels.stats.diagnostic as dg
import statsmodels.tools.tools as ct

extra_var = pd.read_csv('operating_profit.csv')
extra_var = extra_var.drop_duplicates(subset=['Instrument', 'Date'])
refinitivdata = pd.read_csv('refinitivdata4.csv')
assetsusd = pd.read_csv('assetsusd.csv')
new_column_names = {'Instrument': 'Instrument', 'Total Assets': 'Total Assets USD', 'Date': 'Date'}
assetsusd = assetsusd.rename(columns=new_column_names)
assetsusd = assetsusd.drop_duplicates(subset=['Instrument', 'Date'])
refinitivdata = refinitivdata.drop_duplicates(subset=['Instrument', 'Date'])
refinitivdata = pd.merge(refinitivdata, assetsusd, on=['Instrument', 'Date'])
refinitivdata = pd.merge(refinitivdata, extra_var, on=['Instrument', 'Date'])
# fixing the dataframe
refinitivdata[['Company Common Name', 'NAICS Sector Code',
               'NAICS Subsector Code', 'NAICS National Industry Code',
               'NAICS Industry Group Code',
               'Country of Exchange', 'Market Capitalization', 'Net Income after Tax',
               "Total Shareholders' Equity incl Minority Intr & Hybrid Debt"]] = refinitivdata.groupby('Instrument')[
    ['Company Common Name', 'NAICS Sector Code',
     'NAICS Subsector Code', 'NAICS National Industry Code',
     'NAICS Industry Group Code',
     'Country of Exchange', 'Market Capitalization', 'Net Income after Tax',
     "Total Shareholders' Equity incl Minority Intr & Hybrid Debt"]].fillna(method='ffill')

# merge data with extra data we needed

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
refinitivdata['Profitability'] = refinitivdata['Operating Profit before Non-Recurring Income/Expense'] / refinitivdata['Total Assets USD']
refinitivdata['CF Volatility std'] = refinitivdata['Operating Profit before Non-Recurring Income/Expense'].rolling(4).std()
refinitivdata['CF Volatility'] = refinitivdata['CF Volatility std'] / refinitivdata['Total Assets USD']
# Merging the gvisin dataset with the refinitivdata dataset to include gvkey as well as ISIN number
refinitivdata_withgvkey = pd.merge(refinitivdata, gvisin, on='Instrument', how='left')

# merging refinitivdata_withgvkey with capitalstructure_sorted
combined_dataset = pd.merge(refinitivdata_withgvkey, capitalstructure_sorted, left_on=['gvkey', 'Fiscal Year'],
                            right_on=['gvkey', 'year'], how='right')
combined_dataset = combined_dataset[~combined_dataset["NAICS Sector Code"].isin(['22', '52', '53'])]
combined_dataset = combined_dataset[~combined_dataset["Date"].isin(["NaT"])]
combined_dataset = combined_dataset[combined_dataset['Date'].dt.year.isin(range(2001, 2022))]
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
#
combined_dataset['MV Equity'] = combined_dataset['Price Close'] * combined_dataset[
    'Common Shares - ''Outstanding - Total - ''Ord/DR/CPO']
combined_dataset['Market Leverage'] = combined_dataset['Debt - Total'] / (combined_dataset['Debt - Total'] + combined_dataset['MV Equity'])


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
combined_dataset['TD'] = combined_dataset[
    ["Term Loans", "Bonds and Notes", "Revolving Credit", "Other Borrowings", "Capital Lease", "Commercial Paper",
     "Trust Preferred"]].sum(axis=1)

combined_dataset['SS'] = (combined_dataset['Revolving Credit'] / combined_dataset['TD']) ** 2 + (
        combined_dataset['Term Loans'] / combined_dataset['TD']) ** 2 + \
                         (combined_dataset['Bonds and Notes'] / combined_dataset['TD']) ** 2 + (
                                 combined_dataset['Capital Lease'] / combined_dataset['TD']) ** 2 + \
                         (combined_dataset['Commercial Paper'] / combined_dataset['TD']) ** 2 + \
                         (combined_dataset['Other Borrowings'] / combined_dataset['TD']) ** 2
combined_dataset['HHI'] = (combined_dataset['SS'] - (1 / 6)) / (1 - (1 / 6))

### HHI aggregated on years ###
HHI_annual = combined_dataset[['HHI', 'year']].groupby(['year']).mean().transpose()


### Debt specialization dummy ###
def ds90(df):
    df['DS90 dummy'] = 0
    df.loc[(df['Revolving Credit/Total Debt'] >= 0.9) | (df['Bonds and Notes/Total Debt'] >= 0.9) | (
            df['Term Loans/Total Debt'] >= 0.9) | (df['Commercial Paper/Total Debt'] >= 0.9) | (
                   df['Other Borrowings/Total Debt'] >= 0.9) | (df['Capital Lease/Total Debt'] >= 0.9) | (
                   df['Trust Preferred/Total Debt'] >= 0.9), 'DS90 dummy'] = 1


ds90(combined_dataset)

## annualizing ds90 ##
DS90_annual = combined_dataset[['DS90 dummy', 'year']].groupby('year').mean().transpose()

## combining DS90, HHI_annual and Debt specs into 1 dataframe ##
debttypes_and_debtspecs_over_time = pd.concat([debt_specialization_polar, HHI_annual, DS90_annual]).drop('Total',
                                                                                                         axis=0)
#print(debttypes_and_debtspecs_over_time.to_latex())
## time-varying effect per country
debttypes_and_debtspecs_over_time_norway = combined_dataset[
    ['year', 'Revolving Credit/Total Debt', 'Term Loans/Total Debt', 'Bonds and Notes/Total Debt',
     'Commercial Paper/Total Debt', 'Capital Lease/Total Debt', 'Other Borrowings/Total Debt',
     'Country of Exchange', 'HHI', 'DS90 dummy']]
debttypes_and_debtspecs_over_time_norway = debttypes_and_debtspecs_over_time_norway[debttypes_and_debtspecs_over_time_norway['Country of Exchange']== 'Norway']
debttypes_and_debtspecs_over_time_norway = debttypes_and_debtspecs_over_time_norway.groupby('year').mean().transpose()

debttypes_and_debtspecs_over_time_sweden = combined_dataset[
    ['year', 'Revolving Credit/Total Debt', 'Term Loans/Total Debt', 'Bonds and Notes/Total Debt',
     'Commercial Paper/Total Debt', 'Capital Lease/Total Debt', 'Other Borrowings/Total Debt',
     'Country of Exchange', 'HHI', 'DS90 dummy']]
debttypes_and_debtspecs_over_time_sweden = debttypes_and_debtspecs_over_time_sweden[debttypes_and_debtspecs_over_time_sweden['Country of Exchange']== 'Sweden']
debttypes_and_debtspecs_over_time_sweden = debttypes_and_debtspecs_over_time_sweden.groupby('year').mean().transpose()

debttypes_and_debtspecs_over_time_denmark = combined_dataset[
    ['year', 'Revolving Credit/Total Debt', 'Term Loans/Total Debt', 'Bonds and Notes/Total Debt',
     'Commercial Paper/Total Debt', 'Capital Lease/Total Debt', 'Other Borrowings/Total Debt',
     'Country of Exchange', 'HHI', 'DS90 dummy']]
debttypes_and_debtspecs_over_time_denmark = debttypes_and_debtspecs_over_time_denmark[debttypes_and_debtspecs_over_time_denmark['Country of Exchange']== 'Denmark']
debttypes_and_debtspecs_over_time_denmark = debttypes_and_debtspecs_over_time_denmark.groupby('year').mean().transpose()

debttypes_and_debtspecs_over_time_Finland = combined_dataset[
    ['year', 'Revolving Credit/Total Debt', 'Term Loans/Total Debt', 'Bonds and Notes/Total Debt',
     'Commercial Paper/Total Debt', 'Capital Lease/Total Debt', 'Other Borrowings/Total Debt',
     'Country of Exchange', 'HHI', 'DS90 dummy']]
debttypes_and_debtspecs_over_time_Finland = debttypes_and_debtspecs_over_time_Finland[debttypes_and_debtspecs_over_time_Finland['Country of Exchange']== 'Finland']
debttypes_and_debtspecs_over_time_Finland = debttypes_and_debtspecs_over_time_Finland.groupby('year').mean().transpose()

print(debttypes_and_debtspecs_over_time_norway.to_latex())
print(debttypes_and_debtspecs_over_time_sweden.to_latex())
print(debttypes_and_debtspecs_over_time_denmark.to_latex())
print(debttypes_and_debtspecs_over_time_Finland.to_latex())
## CLUSTER ANALYSIS ##
scatterdata = combined_dataset.copy()
scatterdata['Other Borrowings/Total Debt'] = scatterdata['Other Borrowings/Total Debt'] + scatterdata[
    'Trust Preferred/Total Debt']

## creating a subplot of scatterdata to find patterns for the clusters
clusterpatterns = scatterdata[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                               'Other Borrowings/Total Debt', 'Capital Lease/Total Debt',
                               'Commercial Paper/Total Debt']]

clusterpatterns = clusterpatterns.replace([np.inf, -np.inf], np.nan).fillna(0)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(clusterpatterns)
print(clusterpatterns.describe())

kmeans = KMeans(n_clusters=2, init='k-means++')
kmeans.fit(data_scaled)
kmeans.inertia_

SSE = []
for cluster in range(1, 20):
    kmeans = KMeans(n_clusters=cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

frame = pd.DataFrame({'Cluster': range(1, 20), 'SSE': SSE})
p.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.pyplot.show()

kmeans = KMeans(n_clusters=6, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

frame = pd.DataFrame(data_scaled)
frame['Cluster'] = pred
print(frame['Cluster'].value_counts())

scatterdata.reset_index(inplace=True)
scatterdata['clusters'] = frame['Cluster']

## Include only Scatterdata that has observations for Common Shares, Total Assets, Current Assets, Current Liabilities
scatterdata = scatterdata[scatterdata['Total Current Liabilities'] != 0]
scatterdata = scatterdata[scatterdata['Common Shares - ''Outstanding - Total - ''Ord/DR/CPO'] != 0]
scatterdata = scatterdata[scatterdata['Total Current Assets'] != 0]

### importing a column from a cleaned dataset which is further down in this file
trengerenkolonnebare = pd.read_csv('which_firms_specialize.csv')
trengerenkolonnebare.fillna(0, inplace=True)
trengerenkolonnebare.rename(columns={'Debt - Total': 'Debt - Total USD', 'Market Capitalization': 'Market Cap USD',
                                       'Price Close': 'Price Close USD'}, inplace=True)
trengerenkolonnebare = trengerenkolonnebare.drop_duplicates(subset=['Instrument', 'Date'])
trengerenkolonnebare['Date'] = pd.to_datetime(trengerenkolonnebare['Date'], format='%Y/%m/%d')
trengerenkolonnebare = pd.merge(refinitivdata, trengerenkolonnebare, on=['Instrument', 'Date'])
trengerenkolonnebare = trengerenkolonnebare.fillna(0)
trengerenkolonnebare['Revenue with constant'] = trengerenkolonnebare['Revenue from Business Activities - Total'] + 1
trengerenkolonnebare['Revenue with constant'] = trengerenkolonnebare['Revenue with constant'][
    trengerenkolonnebare['Revenue with constant'] > 0]
trengerenkolonnebare['Assets with constant'] = trengerenkolonnebare['Total Assets USD'] + 1
trengerenkolonnebare['Assets with constant'] = trengerenkolonnebare['Assets with constant'][
    trengerenkolonnebare['Assets with constant'] > 0]
trengerenkolonnebare['Debt - Total USD'] = trengerenkolonnebare['Debt - Total USD'][trengerenkolonnebare['Debt - Total USD'] > 0]
trengerenkolonnebare['Total Assets USD'] = trengerenkolonnebare['Total Assets USD'][trengerenkolonnebare['Total Assets USD'] > 0]

trengerenkolonnebare['Market Cap USD'] = trengerenkolonnebare['Market Cap USD'].drop_duplicates()
trengerenkolonnebare['Net Income after Tax'] = trengerenkolonnebare['Net Income after Tax'].drop_duplicates()
trengerenkolonnebare["Total Shareholders' Equity incl Minority Intr & Hybrid Debt"] = trengerenkolonnebare[
    "Total Shareholders' Equity incl Minority Intr & Hybrid Debt"].drop_duplicates()
trengerenkolonnebare['Date'] = pd.to_datetime(trengerenkolonnebare['Date'])
trengerenkolonnebare = trengerenkolonnebare[trengerenkolonnebare['Date'].dt.year.isin(range(2001, 2022))]

### adding Market Leverage, Liquidity and Size to  the scatterdata dataset
#scatterdata['MV Equity'] = scatterdata['Price Close'] * scatterdata[
#    'Common Shares - ''Outstanding - Total - ''Ord/DR/CPO']
scatterdata['Book Leverage'] = trengerenkolonnebare['Debt - Total USD'] / trengerenkolonnebare['Total Assets USD']
#scatterdata['Market Leverage'] = scatterdata['Debt - Total'] / (scatterdata['Debt - Total'] + scatterdata['MV Equity'])
scatterdata['Liquidity'] = scatterdata['Total Current Assets'] / scatterdata['Total Current Liabilities']
scatterdata['Size Assets USD'] = scatterdata['Total Assets USD']
scatterdata['Size Mcap USD'] = trengerenkolonnebare['Market Cap USD']
scatterdata['M/B'] = trengerenkolonnebare['Market Cap USD'] / trengerenkolonnebare['Total Assets USD']
scatterdata['CF Volatility'] = combined_dataset['CF Volatility']
#### convert to polar ####
scatterdata_polar = scatterdata[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Other Borrowings/Total Debt', 'Capital Lease/Total Debt',
                                 'Commercial Paper/Total Debt', 'clusters', 'HHI',
                                 'Profitability', 'MV Equity', 'Market Leverage', 'Liquidity', 'Size Mcap USD', 'Size Assets USD', 'M/B', 'CF Volatility']]
scatterdata_polar = pl.from_pandas(scatterdata_polar[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt',
                                                      'Revolving Credit/Total Debt', 'Other Borrowings/Total Debt',
                                                      'Capital Lease/Total Debt',
                                                      'Commercial Paper/Total Debt', 'clusters', 'HHI',
                                                      'Profitability', 'MV Equity', 'Market Leverage', 'Liquidity', 'Size Mcap USD', 'Size Assets USD', 'M/B', 'CF Volatility']])
### observasjoner av de ulike clusterne
print((scatterdata['Term Loans/Total Debt'] != 0).sum())
print((scatterdata['Bonds and Notes/Total Debt'] != 0).sum())
print((scatterdata['Revolving Credit/Total Debt'] != 0).sum())
print((scatterdata['Other Borrowings/Total Debt'] != 0).sum())
print((scatterdata['Capital Lease/Total Debt'] != 0).sum())
print((scatterdata['Commercial Paper/Total Debt'] != 0).sum())


scatterdata_polar = scatterdata_polar.groupby(
    [
        'clusters'
    ]
).agg(
    [
        pl.mean('Term Loans/Total Debt'),
        pl.mean('Bonds and Notes/Total Debt'),
        pl.mean('Revolving Credit/Total Debt'),
        pl.mean('Other Borrowings/Total Debt'),
        pl.mean('Capital Lease/Total Debt'),
        pl.mean('Commercial Paper/Total Debt'),
        pl.mean('HHI'),
        pl.mean('Profitability'),
        pl.mean('Market Leverage'),
        pl.mean('Liquidity'),
        pl.mean('Size Mcap USD'),
        pl.mean('Size Assets USD'),
        pl.mean('M/B'),
        pl.mean('CF Volatility')
    ]
).to_pandas()
scatterdata_polar.set_index('clusters', inplace=True)
scatterdata_polar.sort_index(ascending=True, inplace=True)

##
print(scatterdata_polar.to_latex())
#print(latex_cluster_table)

datafor3d = scatterdata_polar[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                               'Other Borrowings/Total Debt', 'Capital Lease/Total Debt',
                               'Commercial Paper/Total Debt']]
##### prøver å lage 3d chart av clusteringen
result = np.array(datafor3d)
colors = ['r', 'b', 'g', 'y', 'b', 'p']
fig = p.figure(figsize=(8, 8), dpi=250)
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_ylabel('Cluster', labelpad=10)
ax1.set_zlabel('Percentage DS')
xlabels = np.array(['Term Loans', 'Bonds and Notes', 'Revolving Credit',
                    'Other Borrowings', 'Capital Lease', 'Commercial Paper'])
xpos = np.arange(xlabels.shape[0])
ylabels = np.array(['0', '1', '2', '3', '4', '5'])
ypos = np.arange(ylabels.shape[0])

xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

zpos = result
zpos = zpos.ravel()

dx = 0.5
dy = 0.5
dz = zpos

ax1.xaxis.set_ticks(xpos + dx / 2.)
plt.pyplot.xticks(rotation=45)
ax1.xaxis.set_ticklabels(xlabels)

ax1.yaxis.set_ticks(ypos + dy / 2.)
ax1.yaxis.set_ticklabels(ylabels)

values = np.linspace(0.2, 1., xposM.ravel().shape[0])
colors = cm.rainbow(values)
ax1.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors)
p.show()

## Table for concentration of debt specialization
debtconcentrationdf = combined_dataset.copy()
debtconcentrationdf['Other Borrowings/Total Debt'] = debtconcentrationdf['Other Borrowings/Total Debt'] + \
                                                     debtconcentrationdf['Trust Preferred/Total Debt']
debtconcentrationdf = debtconcentrationdf[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
percentages_df = pd.DataFrame()

for threshold in thresholds:
    threshold_dict = {}
    for col in debtconcentrationdf.columns:
        percentage = (debtconcentrationdf[col] >= threshold).mean() * 100
        threshold_dict[f'{col}_percentage'] = percentage
    percentages_df = percentages_df.append(threshold_dict, ignore_index=True)

percentages_df = percentages_df.transpose()
print(percentages_df.to_latex())
## Debtconcentration for each country individually
# NORWAY #
debtconcentration_norway = combined_dataset.copy()
debtconcentration_norway = debtconcentration_norway[debtconcentration_norway['Country of Exchange'] == 'Norway']
debtconcentration_norway['Other Borrowings/Total Debt'] = debtconcentration_norway['Other Borrowings/Total Debt'] + \
                                                          debtconcentration_norway['Trust Preferred/Total Debt']
debtconcentration_norway = debtconcentration_norway[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
percentages_norway_df = pd.DataFrame()

for threshold in thresholds:
    threshold_dict = {}
    for col in debtconcentration_norway.columns:
        percentage = (debtconcentration_norway[col] >= threshold).mean() * 100
        threshold_dict[f'{col}_percentage'] = percentage
    percentages_norway_df = percentages_norway_df.append(threshold_dict, ignore_index=True)

percentages_norway_df = percentages_norway_df.transpose()

# SWEDEN #
debtconcentration_sweden = combined_dataset.copy()
debtconcentration_sweden = debtconcentration_sweden[debtconcentration_sweden['Country of Exchange'] == 'Sweden']
debtconcentration_sweden['Other Borrowings/Total Debt'] = debtconcentration_sweden['Other Borrowings/Total Debt'] + \
                                                          debtconcentration_sweden['Trust Preferred/Total Debt']
debtconcentration_sweden = debtconcentration_sweden[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
percentages_sweden_df = pd.DataFrame()

for threshold in thresholds:
    threshold_dict = {}
    for col in debtconcentration_sweden.columns:
        percentage = (debtconcentration_sweden[col] >= threshold).mean() * 100
        threshold_dict[f'{col}_percentage'] = percentage
    percentages_sweden_df = percentages_sweden_df.append(threshold_dict, ignore_index=True)

percentages_sweden_df = percentages_sweden_df.transpose()

# DENMARK #
debtconcentration_denmark = combined_dataset.copy()
debtconcentration_denmark = debtconcentration_denmark[debtconcentration_denmark['Country of Exchange'] == 'Denmark']
debtconcentration_denmark['Other Borrowings/Total Debt'] = debtconcentration_denmark['Other Borrowings/Total Debt'] + \
                                                           debtconcentration_denmark['Trust Preferred/Total Debt']
debtconcentration_denmark = debtconcentration_denmark[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
percentages_denmark_df = pd.DataFrame()

for threshold in thresholds:
    threshold_dict = {}
    for col in debtconcentration_denmark.columns:
        percentage = (debtconcentration_denmark[col] >= threshold).mean() * 100
        threshold_dict[f'{col}_percentage'] = percentage
    percentages_denmark_df = percentages_denmark_df.append(threshold_dict, ignore_index=True)

percentages_denmark_df = percentages_denmark_df.transpose()

# FINLAND #
debtconcentration_finland = combined_dataset.copy()
debtconcentration_finland = debtconcentration_finland[debtconcentration_finland['Country of Exchange'] == 'Finland']
debtconcentration_finland['Other Borrowings/Total Debt'] = debtconcentration_finland['Other Borrowings/Total Debt'] + \
                                                           debtconcentration_finland['Trust Preferred/Total Debt']
debtconcentration_finland = debtconcentration_finland[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
percentages_finland_df = pd.DataFrame()

for threshold in thresholds:
    threshold_dict = {}
    for col in debtconcentration_finland.columns:
        percentage = (debtconcentration_finland[col] >= threshold).mean() * 100
        threshold_dict[f'{col}_percentage'] = percentage
    percentages_finland_df = percentages_finland_df.append(threshold_dict, ignore_index=True)

percentages_finland_df = percentages_finland_df.transpose()

## conditional debt concentration
conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Term Loans/Total Debt'] >= 0.3]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage = pd.DataFrame()
conditional_debt_concentration_percentage['TL30avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Bonds and Notes/Total Debt'] >= 0.3]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['B&N30avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Revolving Credit/Total Debt'] >= 0.3]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['RC30avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Capital Lease/Total Debt'] >= 0.3]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['CL30avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Commercial Paper/Total Debt'] >= 0.3]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['CP30avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Other Borrowings/Total Debt'] >= 0.3]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['OB30avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration_percentage = conditional_debt_concentration_percentage.transpose()
#print(conditional_debt_concentration_percentage.to_latex())
## conditional debt concentration
conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Term Loans/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage = pd.DataFrame()
conditional_debt_concentration_percentage['TL50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Bonds and Notes/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['B&N50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Revolving Credit/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['RC50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Capital Lease/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['CL50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Commercial Paper/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['CP50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration[
                                                                    'Other Borrowings/Total Debt'] + \
                                                                conditional_debt_concentration[
                                                                    'Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[
    conditional_debt_concentration['Other Borrowings/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['OB50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration_percentage = conditional_debt_concentration_percentage.transpose()
#print(conditional_debt_concentration_percentage.to_latex())
## conditional debt concentration on industry
conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['NAICS Sector Code'].isin(['11'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry[
                                                                             'Other Borrowings/Total Debt'] + \
                                                                         conditional_debt_concentration_industry[
                                                                             'Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['Term Loans/Total Debt'] >= 0.30]

conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage = pd.DataFrame()
conditional_debt_concentration_percentage['TL50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['NAICS Sector Code'].isin(['11'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry[
                                                                             'Other Borrowings/Total Debt'] + \
                                                                         conditional_debt_concentration_industry[
                                                                             'Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['Bonds and Notes/Total Debt'] >= 0.3]
conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage['B&N50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['NAICS Sector Code'].isin(['11'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry[
                                                                             'Other Borrowings/Total Debt'] + \
                                                                         conditional_debt_concentration_industry[
                                                                             'Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['Revolving Credit/Total Debt'] >= 0.3]
conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage['RC50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['NAICS Sector Code'].isin(['11'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry[
                                                                             'Other Borrowings/Total Debt'] + \
                                                                         conditional_debt_concentration_industry[
                                                                             'Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['Capital Lease/Total Debt'] >= 0.3]
conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage['CL50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['NAICS Sector Code'].isin(['11'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry[
                                                                             'Other Borrowings/Total Debt'] + \
                                                                         conditional_debt_concentration_industry[
                                                                             'Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['Commercial Paper/Total Debt'] >= 0.3]
conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage['CP50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['NAICS Sector Code'].isin(['11'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry[
                                                                             'Other Borrowings/Total Debt'] + \
                                                                         conditional_debt_concentration_industry[
                                                                             'Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
     'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[
    conditional_debt_concentration_industry['Other Borrowings/Total Debt'] >= 0.3]
conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage['OB50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_percentage = conditional_debt_concentration_percentage.transpose()

###### conditional debt country specific
conditional_debt_concentration_country = combined_dataset.copy()
for country in conditional_debt_concentration_country['Country of Exchange'].unique():
    globals()[f"conditional_debt_concentration_{country}"] = combined_dataset.copy()
    globals()[f"conditional_debt_concentration_{country}"]['Other Borrowings/Total Debt'] = globals()[f"conditional_debt_concentration_{country}"][
                                                                                'Other Borrowings/Total Debt'] + \
                                                                            globals()[f"conditional_debt_concentration_{country}"][
                                                                                'Trust Preferred/Total Debt']
    globals()[f"conditional_debt_concentration_{country}"] = globals()[f"conditional_debt_concentration_{country}"][globals()[f"conditional_debt_concentration_{country}"]['Country of Exchange'] == country]
    globals()[f"conditional_debt_concentration_{country}"] = globals()[f"conditional_debt_concentration_{country}"][
        ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
         'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']].dropna()
    total = []
    tl30 = globals()[f"conditional_debt_concentration_{country}"][
        globals()[f"conditional_debt_concentration_{country}"]['Term Loans/Total Debt'] >= 0.3]
    tl30count = tl30['Term Loans/Total Debt'].count()
    tl30 = tl30.transpose()
    total.append(tl30count)
    bn30 = globals()[f"conditional_debt_concentration_{country}"][
        globals()[f"conditional_debt_concentration_{country}"]['Bonds and Notes/Total Debt'] >= 0.3]
    bn30count = bn30['Bonds and Notes/Total Debt'].count()
    bn30 = bn30.transpose()
    total.append(bn30count)
    rc30 = globals()[f"conditional_debt_concentration_{country}"][
        globals()[f"conditional_debt_concentration_{country}"]['Revolving Credit/Total Debt'] >= 0.3]
    rc30count = rc30['Revolving Credit/Total Debt'].count()
    rc30 = rc30.transpose()
    total.append(rc30count)
    cl30 = globals()[f"conditional_debt_concentration_{country}"][
        globals()[f"conditional_debt_concentration_{country}"]['Capital Lease/Total Debt'] >= 0.3]
    cl30count = cl30['Capital Lease/Total Debt'].count()
    cl30 = cl30.transpose()
    total.append(cl30count)
    cp30 = globals()[f"conditional_debt_concentration_{country}"][
        globals()[f"conditional_debt_concentration_{country}"]['Commercial Paper/Total Debt'] >= 0.3]
    cp30count = cp30['Commercial Paper/Total Debt'].count()
    cp30 = cp30.transpose()
    total.append(cp30count)
    ob30 = globals()[f"conditional_debt_concentration_{country}"][
        globals()[f"conditional_debt_concentration_{country}"]['Other Borrowings/Total Debt'] >= 0.3]
    ob30count = ob30['Other Borrowings/Total Debt'].count()
    ob30 = ob30.transpose()
    total.append(ob30count)

    globals()[f"conditional_debt_concentration_{country}_percentage"] = pd.DataFrame()
    globals()[f"conditional_debt_concentration_{country}_percentage"]['TL30avg'] = tl30.mean(axis=1)
    globals()[f"conditional_debt_concentration_{country}_percentage"]['B&N30avg'] = bn30.mean(axis=1)
    globals()[f"conditional_debt_concentration_{country}_percentage"]['RC30avg'] = rc30.mean(axis=1)
    globals()[f"conditional_debt_concentration_{country}_percentage"]['CL30avg'] = cl30.mean(axis=1)
    globals()[f"conditional_debt_concentration_{country}_percentage"]['CP30avg'] = cp30.mean(axis=1)
    globals()[f"conditional_debt_concentration_{country}_percentage"]['OB30avg'] = ob30.mean(axis=1)


    globals()[f"conditional_debt_concentration_{country}_percentage"] = globals()[f"conditional_debt_concentration_{country}_percentage"].transpose()
    globals()[f"conditional_debt_concentration_{country}_percentage"]['Observations'] = total

######## conditional debt NAICS specific
conditional_debt_concentration_naics = combined_dataset.copy()
conditional_debt_concentration_naics['NAICS Sector Code'].replace({'31-33': '31_33', '44-45': '44_45', '48-49': '48_49'}, inplace=True)
for naics in conditional_debt_concentration_naics['NAICS Sector Code'].unique():
    globals()[f"conditional_debt_concentration_{naics}"] = combined_dataset.copy()
    globals()[f"conditional_debt_concentration_{naics}"]['NAICS Sector Code'].replace({'31-33': '31_33', '44-45': '44_45', '48-49': '48_49'}, inplace=True)
    globals()[f"conditional_debt_concentration_{naics}"]['Other Borrowings/Total Debt'] = globals()[f"conditional_debt_concentration_{naics}"][
                                                                                'Other Borrowings/Total Debt'] + \
                                                                                          globals()[f"conditional_debt_concentration_{naics}"][
                                                                                'Trust Preferred/Total Debt']
    globals()[f"conditional_debt_concentration_{naics}"] = globals()[f"conditional_debt_concentration_{naics}"][globals()[f"conditional_debt_concentration_{naics}"]['NAICS Sector Code'] == naics]
    globals()[f"conditional_debt_concentration_{naics}"] = globals()[f"conditional_debt_concentration_{naics}"][
        ['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
         'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']].dropna()
    total = []
    tl30 = globals()[f"conditional_debt_concentration_{naics}"][
        globals()[f"conditional_debt_concentration_{naics}"]['Term Loans/Total Debt'] >= 0.3]
    tl30count = tl30['Term Loans/Total Debt'].count()
    tl30 = tl30.transpose()
    total.append(tl30count)
    bn30 = globals()[f"conditional_debt_concentration_{naics}"][
        globals()[f"conditional_debt_concentration_{naics}"]['Bonds and Notes/Total Debt'] >= 0.3]
    bn30count = bn30['Bonds and Notes/Total Debt'].count()
    bn30 = bn30.transpose()
    total.append(bn30count)
    rc30 = globals()[f"conditional_debt_concentration_{naics}"][
        globals()[f"conditional_debt_concentration_{naics}"]['Revolving Credit/Total Debt'] >= 0.3]
    rc30count = rc30['Revolving Credit/Total Debt'].count()
    rc30 = rc30.transpose()
    total.append(rc30count)
    cl30 = globals()[f"conditional_debt_concentration_{naics}"][
        globals()[f"conditional_debt_concentration_{naics}"]['Capital Lease/Total Debt'] >= 0.3]
    cl30count = cl30['Capital Lease/Total Debt'].count()
    cl30 = cl30.transpose()
    total.append(cl30count)
    cp30 = globals()[f"conditional_debt_concentration_{naics}"][
        globals()[f"conditional_debt_concentration_{naics}"]['Commercial Paper/Total Debt'] >= 0.3]
    cp30count = cp30['Commercial Paper/Total Debt'].count()
    cp30 = cp30.transpose()
    total.append(cp30count)
    ob30 = globals()[f"conditional_debt_concentration_{naics}"][
        globals()[f"conditional_debt_concentration_{naics}"]['Other Borrowings/Total Debt'] >= 0.3]
    ob30count = ob30['Other Borrowings/Total Debt'].count()
    ob30 = ob30.transpose()
    total.append(ob30count)

    globals()[f"conditional_debt_concentration_{naics}_percentage"] = pd.DataFrame()
    globals()[f"conditional_debt_concentration_{naics}_percentage"]['TL30avg'] = tl30.mean(axis=1)
    globals()[f"conditional_debt_concentration_{naics}_percentage"]['B&N30avg'] = bn30.mean(axis=1)
    globals()[f"conditional_debt_concentration_{naics}_percentage"]['RC30avg'] = rc30.mean(axis=1)
    globals()[f"conditional_debt_concentration_{naics}_percentage"]['CL30avg'] = cl30.mean(axis=1)
    globals()[f"conditional_debt_concentration_{naics}_percentage"]['CP30avg'] = cp30.mean(axis=1)
    globals()[f"conditional_debt_concentration_{naics}_percentage"]['OB30avg'] = ob30.mean(axis=1)


    globals()[f"conditional_debt_concentration_{naics}_percentage"] = globals()[f"conditional_debt_concentration_{naics}_percentage"].transpose()
    globals()[f"conditional_debt_concentration_{naics}_percentage"]['Observations'] = total

### Debt specialization (Which firms specialize)
which_firms_specialize = pd.read_csv('which_firms_specialize.csv')
which_firms_specialize.fillna(0, inplace=True)
which_firms_specialize.rename(columns={'Debt - Total': 'Debt - Total USD', 'Market Capitalization': 'Market Cap USD',
                                       'Price Close': 'Price Close USD'}, inplace=True)
which_firms_specialize = which_firms_specialize.drop_duplicates(subset=['Instrument', 'Date'])
which_firms_specialize['Date'] = pd.to_datetime(which_firms_specialize['Date'], format='%Y/%m/%d')
which_firms_specialize = pd.merge(refinitivdata, which_firms_specialize, on=['Instrument', 'Date'])
which_firms_specialize = which_firms_specialize.fillna(0)
which_firms_specialize['Revenue with constant'] = which_firms_specialize['Revenue from Business Activities - Total'] + 1
which_firms_specialize['Revenue with constant'] = which_firms_specialize['Revenue with constant'][
    which_firms_specialize['Revenue with constant'] > 0]
which_firms_specialize['Assets with constant'] = which_firms_specialize['Total Assets USD'] + 1
which_firms_specialize['Assets with constant'] = which_firms_specialize['Assets with constant'][
    which_firms_specialize['Assets with constant'] > 0]
which_firms_specialize['CF Volatility'] = combined_dataset['CF Volatility']

which_firms_specialize['Market Cap USD'] = which_firms_specialize['Market Cap USD'].drop_duplicates()
which_firms_specialize['Net Income after Tax'] = which_firms_specialize['Net Income after Tax'].drop_duplicates()
which_firms_specialize["Total Shareholders' Equity incl Minority Intr & Hybrid Debt"] = which_firms_specialize[
    "Total Shareholders' Equity incl Minority Intr & Hybrid Debt"].drop_duplicates()
which_firms_specialize['Date'] = pd.to_datetime(which_firms_specialize['Date'])
which_firms_specialize = which_firms_specialize[which_firms_specialize['Date'].dt.year.isin(range(2001, 2022))]

spec_data_needed = pd.DataFrame()
spec_data_needed['ln Size'] = np.log(which_firms_specialize['Assets with constant'])
spec_data_needed['ln Sales'] = np.log(which_firms_specialize['Revenue with constant'])
spec_data_needed['M/B'] = which_firms_specialize['Market Cap USD'] / which_firms_specialize['Total Assets USD']
spec_data_needed['Profitability'] = which_firms_specialize['Profitability']
spec_data_needed['Dividend Payer'] = which_firms_specialize['Dividend Per Share - Mean'].apply(
    lambda x: 1 if x != 0 else 0)
#spec_data_needed['Cash Holdings'] = which_firms_specialize['Cash & Short Term Investments'] / which_firms_specialize[
#    'Total Assets USD']
spec_data_needed['Tangibility'] = which_firms_specialize['PPE - Net Percentage of Total Assets'] / 100
spec_data_needed['Book Leverage'] = which_firms_specialize['Debt - Total USD'] / which_firms_specialize[
    'Total Assets USD']
#spec_data_needed['CAPEX'] = which_firms_specialize['CAPEX Percentage of Total Assets']
#spec_data_needed['CAPEX'] = spec_data_needed['CAPEX'] / 100
spec_data_needed['Advertising'] = which_firms_specialize['Selling General & Administrative Expenses - Total'] / \
                                  which_firms_specialize['Total Assets USD']
spec_data_needed['Instrument'] = which_firms_specialize['Instrument']
spec_data_needed['Date'] = which_firms_specialize['Date']
spec_data_needed = spec_data_needed.replace([np.inf, -np.inf], np.nan).dropna()

# merging spec_data_needed with HHI and DS90
spec_data_combined = pd.merge(spec_data_needed, combined_dataset, on=['Instrument', 'Date'])
spec_data_combined.dropna(axis=0, inplace=True)

spec_data_needed = spec_data_combined[['ln Size', 'ln Sales', 'M/B', 'Profitability_x', 'Dividend Payer',
                                       'Tangibility','CF Volatility', 'Book Leverage',  'Advertising', 'Instrument', 'Date',
                                       'HHI', 'DS90 dummy', 'Market Leverage']]
# Correlation between variables
rho = spec_data_needed.corr()
pval = spec_data_needed.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))
rho.round(2).astype(str) + p
#print(rho.to_latex())
spec_data_needed_sorted = spec_data_needed.sort_values('HHI')
spec_data_needed_sorted['quartile'] = pd.qcut(spec_data_needed_sorted['HHI'], q=3,
                                              labels=['1st tertile', '2nd tertile', '3rd tertile'])

tertile1 = spec_data_needed['HHI'].quantile(0.33)
tertile2 = spec_data_needed['HHI'].quantile(0.67)

# create new columns based on HHI percentiles
spec_data_needed['1st T'] = (spec_data_needed['HHI'] <= tertile1).astype(int)
spec_data_needed['3rd T'] = (spec_data_needed['HHI'] >= tertile2).astype(int)

mean_median = pd.DataFrame()
mean_median['Mean 1st T'] = spec_data_needed[spec_data_needed['1st T'] == 1].mean()
mean_median['Median 1st T'] = spec_data_needed[spec_data_needed['1st T'] == 1].median()
mean_median['Mean 3rd T'] = spec_data_needed[spec_data_needed['3rd T'] == 1].mean()
mean_median['Median 3rd T'] = spec_data_needed[spec_data_needed['3rd T'] == 1].median()
print(mean_median.to_latex())
# T-test between 1st and 4th quantile
t_test_dataframe = spec_data_needed.drop(['Instrument', 'Date'], axis=1)
exclude_cols = ['DS90 dummy', '1st T', '3rd T', 'HHI']

for col in t_test_dataframe.columns:
    if col in exclude_cols:
        continue
    values1 = t_test_dataframe[t_test_dataframe['1st T'] == 1][col].values
    values2 = t_test_dataframe[t_test_dataframe['3rd T'] == 1][col].values
    t_stat, p_value = ttest_ind(values1, values2, equal_var=False)
    print(f"T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f}")

ss_t_test = t_test_dataframe.describe()

# wilcoxon test
wilcoxon_dataframe = spec_data_needed.drop(['Instrument', 'Date'], axis=1)
exclude_cols = ['DS90 dummy', '1st T', '3rd T', 'HHI']

for col in t_test_dataframe.columns:
    if col in exclude_cols:
        continue
    column1 = wilcoxon_dataframe[wilcoxon_dataframe['1st T'] == 1][col].values
    column2 = wilcoxon_dataframe[wilcoxon_dataframe['3rd T'] == 1][col].values
    statistic, pvalue = stats.wilcoxon(column1, column2)
    n = len(column1)
    W = statistic
    Z = (W - (n * (n + 1)) / 4) / np.sqrt((n * (n + 1) * (2 * n + 1)) / 24)
    print(f" Statistic result = {statistic:.2f}, p-value = {pvalue:.2f}, Z-score = {Z:.2f}")

ss_wilcoxon = wilcoxon_dataframe.describe()

### Multivariate regression
multivariate_reg = spec_data_needed.copy()
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy'], axis=1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date']], how='left',
                            on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)
multivariate_merge['M/B'] = winsorize(multivariate_merge['M/B'], limits=[0.01, 0.01])
multivariate_merge['Profitability_x'] = winsorize(multivariate_merge['Profitability_x'], limits=[0.01, 0.01])
#multivariate_merge['Cash Holdings'] = winsorize(multivariate_merge['Cash Holdings'], limits=[0.01, 0.01])
multivariate_merge['Tangibility'] = winsorize(multivariate_merge['Tangibility'], limits=[0.01, 0.01])
multivariate_merge['Book Leverage'] = winsorize(multivariate_merge['Book Leverage'], limits=[0.01, 0.01])
#multivariate_merge['CAPEX'] = winsorize(multivariate_merge['CAPEX'], limits=[0.01, 0.01])
multivariate_merge['Advertising'] = winsorize(multivariate_merge['Advertising'], limits=[0.01, 0.01])

Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg = sm.OLS(Y, X)
multivariate_reg_res = multivariate_reg.fit(cov_type="HC0")
print(multivariate_reg_res.summary())
print(dg.acorr_breusch_godfrey(multivariate_reg_res, nlags=200))
### adding NAICS dummies
multivariate_reg1 = spec_data_needed.copy()
multivariate_reg1 = multivariate_reg1.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy'], axis=1)
multivariate_reg1 = pd.merge(multivariate_reg1, which_firms_specialize[['NAICS Sector Code', 'Instrument', 'Date']], how='left',
                            on=['Instrument', 'Date'])
multivariate_reg1['Date'] = multivariate_reg1['Date'].dt.year
##lagging
multivariate_reg1.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg1.groupby(level='Instrument').shift(-1)
multivariate_reg1.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge1 = pd.merge(multivariate_reg1, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge1.reset_index(inplace=True)
multivariate_merge1.drop(['HHI_x', 'Instrument', 'Date'], axis=1, inplace=True)
multivariate_merge1.dropna(inplace=True)
dummies_NAICS = pd.get_dummies(multivariate_merge1['NAICS Sector Code'])
multivariate_merge1 = pd.concat([multivariate_merge1, dummies_NAICS], axis=1)
multivariate_merge1.drop(['NAICS Sector Code'], axis=1, inplace=True)
multivariate_merge1['M/B'] = winsorize(multivariate_merge1['M/B'], limits=[0.01, 0.01])
multivariate_merge1['Profitability_x'] = winsorize(multivariate_merge1['Profitability_x'], limits=[0.01, 0.01])
#multivariate_merge1['Cash Holdings'] = winsorize(multivariate_merge1['Cash Holdings'], limits=[0.01, 0.01])
multivariate_merge1['Tangibility'] = winsorize(multivariate_merge1['Tangibility'], limits=[0.01, 0.01])
multivariate_merge1['Book Leverage'] = winsorize(multivariate_merge1['Book Leverage'], limits=[0.01, 0.01])
#multivariate_merge1['CAPEX'] = winsorize(multivariate_merge1['CAPEX'], limits=[0.01, 0.01])
multivariate_merge1['Advertising'] = winsorize(multivariate_merge1['Advertising'], limits=[0.01, 0.01])


Y = multivariate_merge1['HHI_y'].fillna(0)
X = multivariate_merge1.drop(['HHI_y', '62'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg1 = sm.OLS(Y, X)
multivariate_reg_res1 = multivariate_reg1.fit(cov_type="HC0")
print(multivariate_reg_res1.summary())

### only country dummy
multivariate_reg2 = spec_data_needed.copy()
multivariate_reg2 = multivariate_reg2.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy'], axis=1)
multivariate_reg2 = pd.merge(multivariate_reg2, which_firms_specialize[['Instrument', 'Date', 'Country of Exchange']],
                             how='left', on=['Instrument', 'Date'])
multivariate_reg2['Date'] = multivariate_reg2['Date'].dt.year
##lagging
multivariate_reg2.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg2.groupby(level='Instrument').shift(-1)
multivariate_reg2.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge2 = pd.merge(multivariate_reg2, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge2.reset_index(inplace=True)
multivariate_merge2.drop(['HHI_x', 'Instrument', 'Date'], axis=1, inplace=True)
multivariate_merge2.dropna(inplace=True)

dummies_country = pd.get_dummies(multivariate_merge2['Country of Exchange'])
multivariate_merge2 = pd.concat([multivariate_merge2, dummies_country], axis=1)
multivariate_merge2.drop(['Country of Exchange'], axis=1, inplace=True)
multivariate_merge2['M/B'] = winsorize(multivariate_merge2['M/B'], limits=[0.01, 0.01])
multivariate_merge2['Profitability_x'] = winsorize(multivariate_merge2['Profitability_x'], limits=[0.01, 0.01])
#multivariate_merge2['Cash Holdings'] = winsorize(multivariate_merge2['Cash Holdings'], limits=[0.01, 0.01])
multivariate_merge2['Tangibility'] = winsorize(multivariate_merge2['Tangibility'], limits=[0.01, 0.01])
multivariate_merge2['Book Leverage'] = winsorize(multivariate_merge2['Book Leverage'], limits=[0.01, 0.01])
#multivariate_merge2['CAPEX'] = winsorize(multivariate_merge2['CAPEX'], limits=[0.01, 0.01])
multivariate_merge2['Advertising'] = winsorize(multivariate_merge2['Advertising'], limits=[0.01, 0.01])


Y = multivariate_merge2['HHI_y'].fillna(0)
X = multivariate_merge2.drop(['HHI_y', 'Denmark'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg2 = sm.OLS(Y, X)
multivariate_reg_res2 = multivariate_reg2.fit(cov_type="HC0")
print(multivariate_reg_res2.summary())
print(dg.acorr_breusch_godfrey(multivariate_reg_res2, nlags=20))
### year dummies only
multivariate_reg3 = spec_data_needed.copy()
multivariate_reg3 = multivariate_reg3.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy'], axis=1)
multivariate_reg3 = pd.merge(multivariate_reg3, which_firms_specialize[['Instrument', 'Date']], how='left',
                             on=['Instrument', 'Date'])
multivariate_reg3['Date'] = multivariate_reg3['Date'].dt.year
##lagging
multivariate_reg3.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg3.groupby(level='Instrument').shift(-1)
multivariate_reg3.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge3 = pd.merge(multivariate_reg3, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge3.reset_index(inplace=True)
multivariate_merge3.drop(['HHI_x', 'Instrument'], axis=1, inplace=True)
multivariate_merge3.dropna(inplace=True)


dummies_year = pd.get_dummies(multivariate_merge3['Date'])
multivariate_merge3 = pd.concat([multivariate_merge3, dummies_year], axis=1)
multivariate_merge3.drop(['Date'], axis=1, inplace=True)
multivariate_merge3['M/B'] = winsorize(multivariate_merge3['M/B'], limits=[0.01, 0.01])
multivariate_merge3['Profitability_x'] = winsorize(multivariate_merge3['Profitability_x'], limits=[0.01, 0.01])
#multivariate_merge3['Cash Holdings'] = winsorize(multivariate_merge3['Cash Holdings'], limits=[0.01, 0.01])
multivariate_merge3['Tangibility'] = winsorize(multivariate_merge3['Tangibility'], limits=[0.01, 0.01])
multivariate_merge3['Book Leverage'] = winsorize(multivariate_merge3['Book Leverage'], limits=[0.01, 0.01])
#multivariate_merge3['CAPEX'] = winsorize(multivariate_merge3['CAPEX'], limits=[0.01, 0.01])
multivariate_merge3['Advertising'] = winsorize(multivariate_merge3['Advertising'], limits=[0.01, 0.01])


Y = multivariate_merge3['HHI_y'].fillna(0)
X = multivariate_merge3.drop(['HHI_y', 2001], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg3 = sm.OLS(Y, X)
multivariate_reg_res3 = multivariate_reg3.fit(cov_type="HC0")
print(multivariate_reg_res3.summary())
print(dg.acorr_breusch_godfrey(multivariate_reg_res3, nlags=20))

## Multivariate analysis with year and industry fixed effects for HHI
multivariate_reg4 = spec_data_needed.copy()
multivariate_reg4 = multivariate_reg4.drop(['1st T', '3rd T', 'ln Sales', 'DS90 dummy', 'Market Leverage'], axis=1)
multivariate_reg4 = pd.merge(multivariate_reg4, which_firms_specialize[['NAICS Sector Code', 'Instrument', 'Date', 'Country of Exchange']], how='left',
                            on=['Instrument', 'Date'])
multivariate_reg4['Date'] = multivariate_reg4['Date'].dt.year
multivariate_reg4 = multivariate_reg4[multivariate_reg4['Country of Exchange'] != 'Norway']
##lagging
multivariate_reg4.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg4.groupby(level='Instrument').shift(-1)
multivariate_reg4.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge4 = pd.merge(multivariate_reg4, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge4.reset_index(inplace=True)
multivariate_merge4.drop(['HHI_x', 'Instrument'], axis=1, inplace=True)
multivariate_merge4.dropna(inplace=True)
dummies_country = pd.get_dummies(multivariate_merge4['Country of Exchange'])
dummies_NAICS = pd.get_dummies(multivariate_merge4['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge4['Date'])
multivariate_merge4 = pd.concat([multivariate_merge4, dummies_NAICS, dummies_year, dummies_country], axis=1)
multivariate_merge4.drop(['NAICS Sector Code', 'Date', 'Country of Exchange'], axis=1, inplace=True)
multivariate_merge4['M/B'] = winsorize(multivariate_merge4['M/B'], limits=[0.01, 0.01])
multivariate_merge4['Profitability_x'] = winsorize(multivariate_merge4['Profitability_x'], limits=[0.01, 0.01])
multivariate_merge4['Tangibility'] = winsorize(multivariate_merge4['Tangibility'], limits=[0.01, 0.01])
multivariate_merge4['CF Volatility'] = winsorize(multivariate_merge4['CF Volatility'], limits=[0.01, 0.01])
multivariate_merge4['Advertising'] = winsorize(multivariate_merge4['Advertising'], limits=[0.01, 0.01])
multivariate_merge4['Book Leverage'] = winsorize(multivariate_merge4['Book Leverage'], limits=[0.01, 0.01])




Y = multivariate_merge4['HHI_y'].fillna(0)
X = multivariate_merge4.drop(['HHI_y', '62', 2020, 'Finland'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg4 = sm.OLS(Y, X)
multivariate_reg_res4 = multivariate_reg4.fit(cov_type="HC0")
print(multivariate_reg_res4.summary())

## Multivariate analysis with year and industry fixed effects for HHI
multivariate_reg5 = spec_data_needed.copy()
multivariate_reg5 = multivariate_reg5.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy'], axis=1)
multivariate_reg5 = pd.merge(multivariate_reg5, which_firms_specialize[['NAICS Sector Code', 'Instrument', 'Date', 'Country of Exchange']], how='left',
                             on=['Instrument', 'Date'])
multivariate_reg5['Date'] = multivariate_reg5['Date'].dt.year
##lagging
multivariate_reg5.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg5.groupby(level='Instrument').shift(-1)
multivariate_reg5.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge5 = pd.merge(multivariate_reg5, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge5.reset_index(inplace=True)
multivariate_merge5.drop(['HHI_x', 'Instrument'], axis=1, inplace=True)
multivariate_merge5.dropna(inplace=True)
dummies_country = pd.get_dummies(multivariate_merge5['Country of Exchange'])
dummies_NAICS = pd.get_dummies(multivariate_merge5['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge5['Date'])
#multivariate_merge5 = pd.concat([multivariate_merge5, dummies_NAICS, dummies_year], axis=1)
multivariate_merge5 = pd.concat([multivariate_merge5, dummies_NAICS, dummies_year, dummies_country], axis=1)
multivariate_merge5.drop(['NAICS Sector Code', 'Date', 'Country of Exchange','Advertising', 'Book Leverage'], axis=1, inplace=True)
#multivariate_merge5.drop(['NAICS Sector Code', 'Date'], axis=1, inplace=True)
multivariate_merge5['M/B'] = winsorize(multivariate_merge5['M/B'], limits=[0.01, 0.01])
multivariate_merge5['Profitability_x'] = winsorize(multivariate_merge5['Profitability_x'], limits=[0.001, 0.001])
#multivariate_merge5['Cash Holdings'] = winsorize(multivariate_merge5['Cash Holdings'], limits=[0.01, 0.01])
multivariate_merge5['Tangibility'] = winsorize(multivariate_merge5['Tangibility'], limits=[0.001, 0.001])
#multivariate_merge5['Book Leverage'] = winsorize(multivariate_merge5['Book Leverage'], limits=[0.001, 0.001])
#multivariate_merge5['CAPEX'] = winsorize(multivariate_merge5['CAPEX'], limits=[0.001, 0.001])
#multivariate_merge5['Advertising'] = winsorize(multivariate_merge5['Advertising'], limits=[0.001, 0.001])


Y = multivariate_merge5['HHI_y'].fillna(0)
X = multivariate_merge5.drop(['HHI_y', '71', 2020], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg5 = sm.OLS(Y, X)
multivariate_reg_res5 = multivariate_reg5.fit(cov_type="HC0")
print(multivariate_reg_res5.summary())

# count unique observations and their percentages
counts = combined_dataset['NAICS Sector Code'].value_counts()
percentages = counts / len(combined_dataset) * 100

# print the results
for idx, val in enumerate(counts.index):
    count = counts.get(val, default=0)
    percentage = count / len(combined_dataset) * 100
    print(f"{val}: {count} ({percentage:.2f}%)")
################### regression country specific ##############
for country in combined_dataset['Country of Exchange'].unique():
    globals()[f"spec_data_{country}"] = spec_data_combined[['ln Size', 'M/B', 'Profitability_x', 'CF Volatility',
                                              'Tangibility', 'Book Leverage', 'Advertising', 'Instrument',
                                              'Date', 'HHI', 'DS90 dummy', 'Country of Exchange', 'NAICS Sector Code']]
    globals()[f"spec_data_{country}"]['Date'] = globals()[f"spec_data_{country}"]['Date'].dt.year
    globals()[f"spec_data_{country}"] = globals()[f"spec_data_{country}"][globals()[f"spec_data_{country}"]['Country of Exchange'] == country]
    globals()[f"spec_data_{country}"].set_index(['Instrument', 'Date'], inplace=True)
    shifted = globals()[f"spec_data_{country}"].groupby(level='Instrument').shift(-1)
    globals()[f"spec_data_{country}"].join(shifted.rename(columns=lambda x: x + '_lag'))
    globals()[f"spec_data_{country}_merged"] = pd.merge(globals()[f"spec_data_{country}"], shifted['HHI'], left_index=True, right_index=True, how='left')
    globals()[f"spec_data_{country}_merged"].reset_index(inplace=True)
    globals()[f"spec_data_{country}_merged"].drop(['HHI_x', 'DS90 dummy', 'Instrument', 'Country of Exchange'], axis=1, inplace=True)
    globals()[f"spec_data_{country}_merged"].dropna(inplace=True)
    dummies_NAICS = pd.get_dummies(globals()[f"spec_data_{country}_merged"]['NAICS Sector Code'])
    dummies_year = pd.get_dummies(globals()[f"spec_data_{country}_merged"]['Date'])
    globals()[f"spec_data_{country}_merged"] = pd.concat([globals()[f"spec_data_{country}_merged"], dummies_NAICS, dummies_year], axis=1)
    globals()[f"spec_data_{country}_merged"].drop(['NAICS Sector Code', 'Date'], axis=1, inplace=True)
    globals()[f"spec_data_{country}_merged"].dropna(inplace=True)
    globals()[f"spec_data_{country}_merged"]['M/B'] = winsorize(globals()[f"spec_data_{country}_merged"]['M/B'], limits=[0.01, 0.01])
    globals()[f"spec_data_{country}_merged"]['Profitability_x'] = winsorize(globals()[f"spec_data_{country}_merged"]['Profitability_x'], limits=[0.001, 0.001])
    globals()[f"spec_data_{country}_merged"]['CF Volatility'] = winsorize(globals()[f"spec_data_{country}_merged"]['CF Volatility'], limits=[0.01, 0.01])
    globals()[f"spec_data_{country}_merged"]['Tangibility'] = winsorize(globals()[f"spec_data_{country}_merged"]['Tangibility'], limits=[0.001, 0.001])
    globals()[f"spec_data_{country}_merged"]['Book Leverage'] = winsorize(globals()[f"spec_data_{country}_merged"]['Book Leverage'], limits=[0.001, 0.001])
    #globals()[f"spec_data_{country}_merged"]['CAPEX'] = winsorize(globals()[f"spec_data_{country}_merged"]['CAPEX'], limits=[0.001, 0.001])
    globals()[f"spec_data_{country}_merged"]['Advertising'] = winsorize(globals()[f"spec_data_{country}_merged"]['Advertising'], limits=[0.001, 0.001])

    Y = globals()[f"spec_data_{country}_merged"]['HHI_y']
    X = globals()[f"spec_data_{country}_merged"].drop(globals()[f"spec_data_{country}_merged"].columns[[7, 8, -1]], axis=1)
    X = sm.add_constant(X)
    globals()[f"spec_reg_{country}"] = sm.OLS(Y, X).fit(cov_type="HC0")
    print(f"Regression for {country}")
    print(globals()[f"spec_reg_{country}"].summary())

# checking VIF for country specific regression in the same loop
    Y = globals()[f"spec_data_{country}_merged"]['HHI_y']
    X = globals()[f"spec_data_{country}_merged"].drop(globals()[f"spec_data_{country}_merged"].columns[[8, 9, -1]], axis=1)


    scaler = StandardScaler()
    X.columns = X.columns.astype(str)
    X_std = scaler.fit_transform(X)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X_std, i) for i in range(X_std.shape[1])]
    vif["features"] = X.columns
    print(f"VIF for {country}{vif}")

###
#rho = dummies_NAICS.corr()
#pval = dummies_NAICS.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
#p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))
#rho.round(2).astype(str) + p
#### regressing country vs country ####
norway = spec_data_Norway.copy()
sweden = spec_data_Sweden.copy()
finland = spec_data_Finland.copy()
denmark = spec_data_Denmark.copy()
countries = [norway, sweden, finland, denmark]
for country in countries:
    country.drop(['HHI', 'DS90 dummy', 'Country of Exchange', 'NAICS Sector Code'], axis=1, inplace=True)
mean_median_countries = pd.DataFrame()
mean_median_countries['Mean Norway'] = norway.mean()
mean_median_countries['Median Norway'] = norway.median()
mean_median_countries['Mean Sweden'] = sweden.mean()
mean_median_countries['Median Sweden'] = sweden.median()
mean_median_countries['Mean Finland'] = finland.mean()
mean_median_countries['Median Finland'] = finland.median()
mean_median_countries['Mean Denmark'] = denmark.mean()
mean_median_countries['Median Denmark'] = denmark.median()

# T-test Country vs Country
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Norway, Sweden, equal_var=False)
    print(f"Norway vs Sweden: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Norway, Finland, equal_var=False)
    print(f"Norway vs Finland: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Norway, Denmark, equal_var=False)
    print(f"Norway vs Denmark: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Sweden, Norway, equal_var=False)
    print(f"Sweden vs Norway: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Sweden, Finland, equal_var=False)
    print(f"Sweden vs Finland: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Sweden, Denmark, equal_var=False)
    print(f"Sweden vs Denmark: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Finland, Norway, equal_var=False)
    print(f"Finland vs Norway: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Finland, Sweden, equal_var=False)
    print(f"Finland vs Sweden: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Finland, Denmark, equal_var=False)
    print(f"Finland vs Denmark: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Denmark, Norway, equal_var=False)
    print(f"Denmark vs Norway: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Denmark, Sweden, equal_var=False)
    print(f"Denmark vs Sweden: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")
for col in norway.columns:
    Norway = norway[col].values
    Sweden = sweden[col].values
    Finland = finland[col].values
    Denmark = denmark[col].values
    t_stat, p_value = ttest_ind(Denmark, Finland, equal_var=False)
    print(f"Denmark vs Finland: T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.2f} Variable:{col}")


### Industry differentiated debt specialization with threshold
industry_debt_specialization = combined_dataset.copy()
#thresholds = [0.3, 0.6, 0.9, 0.99]
thresholds = [0.99]
percentages_industry_debt_specialization_naics = pd.DataFrame()
exclude_naics = ['55', '81', '61']
for naics in industry_debt_specialization['NAICS Sector Code'].unique():
    if naics in exclude_naics:
        continue
    industry_debt_specialization_naics = industry_debt_specialization[industry_debt_specialization['NAICS Sector Code'] == naics]
    industry_debt_specialization_naics['Other Borrowings/Total Debt'] = industry_debt_specialization_naics['Other Borrowings/Total Debt'] + industry_debt_specialization_naics['Trust Preferred/Total Debt']
    industry_debt_specialization_naics = industry_debt_specialization_naics.drop('Trust Preferred/Total Debt', axis=1)
    industry_debt_specialization_naics = industry_debt_specialization_naics[['Revolving Credit/Total Debt',
                                                                        'Term Loans/Total Debt',
                                                                        'Bonds and Notes/Total Debt',
                                                                        'Commercial Paper/Total Debt',
                                                                        'Capital Lease/Total Debt',
                                                                        'Other Borrowings/Total Debt']].dropna()


    for threshold in thresholds:
        threshold_dict = {}
        for col in industry_debt_specialization_naics.columns:
            percentage = (industry_debt_specialization_naics[col] >= threshold).mean() * 100
            threshold_dict[f'{col}_percentage'] = percentage
        percentages_industry_debt_specialization_naics = percentages_industry_debt_specialization_naics.append(threshold_dict, ignore_index=True)
print(percentages_industry_debt_specialization_naics.to_latex())



### Country differentiated debt specialization with threshold
industry_debt_specialization = combined_dataset.copy()
industry_debt_specialization = industry_debt_specialization[industry_debt_specialization['Country of Exchange'] == 'Norway']
industry_debt_specialization['Other Borrowings/Total Debt'] = industry_debt_specialization['Other Borrowings/Total Debt'] + industry_debt_specialization['Trust Preferred/Total Debt']
industry_debt_specialization = industry_debt_specialization.drop('Trust Preferred/Total Debt', axis=1)
industry_debt_specialization_norway = industry_debt_specialization[['Revolving Credit/Total Debt',
                                                             'Term Loans/Total Debt', 'Bonds and Notes/Total Debt',
                                                             'Commercial Paper/Total Debt', 'Capital Lease/Total Debt',
                                                             'Other Borrowings/Total Debt']].dropna()


thresholds = [0.3, 0.6, 0.9, 0.99]
percentages_industry_debt_specialization_norway = pd.DataFrame()
industry_debt_specialization = combined_dataset.copy()
country_df_names = {
    'Norway': 'industry_debt_specialization_norway',
    'Sweden': 'industry_debt_specialization_sweden',
    'Finland': 'industry_debt_specialization_finland',
    'Denmark': 'industry_debt_specialization_denmark'
}
for country in industry_debt_specialization['Country of Exchange'].unique():
    df_name = country_df_names[country]
    industry_debt_specialization_country = industry_debt_specialization[industry_debt_specialization['Country of Exchange'] == country]
    industry_debt_specialization_country['Other Borrowings/Total Debt'] = industry_debt_specialization_country['Other Borrowings/Total Debt'] + industry_debt_specialization_country['Trust Preferred/Total Debt']
    industry_debt_specialization_country = industry_debt_specialization_country.drop('Trust Preferred/Total Debt', axis=1)
    industry_debt_specialization_country = industry_debt_specialization_country[['Revolving Credit/Total Debt',
                                                                        'Term Loans/Total Debt',
                                                                        'Bonds and Notes/Total Debt',
                                                                        'Commercial Paper/Total Debt',
                                                                        'Capital Lease/Total Debt',
                                                                        'Other Borrowings/Total Debt']].dropna()
    locals()[df_name] = industry_debt_specialization_country
    for threshold in thresholds:
     threshold_dict = {}
        for col in locals()[df_name].columns:
           percentage = (locals()[df_name][col] >= threshold).mean() * 100
           threshold_dict[f'{col}_percentage'] = percentage
        percentages_industry_debt_specialization_norway = percentages_industry_debt_specialization_norway.append(threshold_dict, ignore_index=True)

### Industry differentiated table with observations and country %
ind_country_table = pd.DataFrame()
ind_country_table['Total observations'] = combined_dataset['NAICS Sector Code'].value_counts()
ind_country_table['% Norway'] = combined_dataset[combined_dataset['Country of Exchange'] == 'Norway'][
                                    'NAICS Sector Code'].value_counts() / ind_country_table['Total observations']
ind_country_table['% Sweden'] = combined_dataset[combined_dataset['Country of Exchange'] == 'Sweden'][
                                    'NAICS Sector Code'].value_counts() / ind_country_table['Total observations']
ind_country_table['% Denmark'] = combined_dataset[combined_dataset['Country of Exchange'] == 'Denmark'][
                                     'NAICS Sector Code'].value_counts() / ind_country_table['Total observations']
ind_country_table['% Finland'] = combined_dataset[combined_dataset['Country of Exchange'] == 'Finland'][
                                     'NAICS Sector Code'].value_counts() / ind_country_table['Total observations']
ind_country_table.fillna(0, inplace=True)
print(ind_country_table.to_latex())
####### tabell med avg av naicskoders hhi
enavgtabell = combined_dataset[['NAICS Sector Code', 'HHI']]
#enavgtabell['NAICS Sector Code'] = enavgtabell['NAICS Sector Code'].apply(lambda x: (int(x)))
enavgtabell = enavgtabell.groupby('NAICS Sector Code').agg({'HHI': 'mean'}).reset_index()
grouped_counts = combined_dataset.groupby('NAICS Sector Code').size().reset_index(name='Observations')
enavgtabell = enavgtabell.merge(grouped_counts, on='NAICS Sector Code')

## samme for flere naics numre
enavgtabell = combined_dataset[['NAICS Industry Group Code', 'HHI']]
enavgtabell['NAICS Industry Group Code'] = enavgtabell['NAICS Industry Group Code'].apply(lambda x: (int(x)))
enavgtabell = enavgtabell.groupby('NAICS Industry Group Code').agg({'HHI': 'mean'}).reset_index()
grouped_counts = combined_dataset.groupby('NAICS Industry Group Code').size().reset_index(name='Observations')
enavgtabell = enavgtabell.merge(grouped_counts, on='NAICS Industry Group Code')

######## summary statistics tabell for variablene
summary_statistics = spec_data_combined[['ln Size', 'ln Sales', 'M/B', 'Profitability_x', 'Dividend Payer', 'CF Volatility', 'Tangibility', 'Book Leverage', 'Advertising', 'Market Leverage']]
summary_statistics['M/B'] = winsorize(summary_statistics['M/B'], limits=[0.01, 0.01])
summary_statistics['Profitability_x'] = winsorize(summary_statistics['Profitability_x'], limits=[0.01, 0.01])
summary_statistics['CF Volatility'] = winsorize(summary_statistics['CF Volatility'], limits=[0.01, 0.01])
summary_statistics['Tangibility'] = winsorize(summary_statistics['Tangibility'], limits=[0.01, 0.01])
summary_statistics['Book Leverage'] = winsorize(summary_statistics['Book Leverage'], limits=[0.01, 0.01])
summary_statistics['Market Leverage'] = winsorize(summary_statistics['Market Leverage'], limits=[0.01, 0.01])

#summary_statistics['CAPEX'] = winsorize(summary_statistics['CAPEX'], limits=[0.001, 0.001])
summary_statistics['Advertising'] = winsorize(summary_statistics['Advertising'], limits=[0.01, 0.01])
summary_statistics = summary_statistics.transpose()

summary_statistics_table = pd.DataFrame()
summary_statistics_table['Mean'] = summary_statistics.mean(axis=1)
summary_statistics_table['Median'] = summary_statistics.median(axis=1)
summary_statistics_table['SE'] = summary_statistics.std(axis=1)
summary_statistics_table['25% Lowest'] = summary_statistics.quantile(0.25, axis=1)
summary_statistics_table['75% Highest'] = summary_statistics.quantile(0.75, axis=1)
summary_statistics_table['Min'] = summary_statistics.min(axis=1)
summary_statistics_table['Max'] = summary_statistics.max(axis=1)
print(summary_statistics_table.to_latex())
## summary statistic for all countries with variables

ss_allcountries_table_df = pd.DataFrame()
for country in spec_data_combined['Country of Exchange'].unique():
    ss_allcountries_table = []
    summary_statistics_allcountries_table = pd.DataFrame()
    summary_statistics_allcountries = spec_data_combined[spec_data_combined['Country of Exchange'] == country][['ln Size', 'ln Sales', 'M/B', 'Profitability_x', 'Dividend Payer', 'CF Volatility', 'Tangibility', 'Book Leverage', 'Advertising']]
    summary_statistics_allcountries['M/B'] = winsorize(summary_statistics_allcountries['M/B'], limits=[0.01, 0.01])
    summary_statistics_allcountries['Profitability_x'] = winsorize(summary_statistics_allcountries['Profitability_x'], limits=[0.01, 0.01])
    summary_statistics_allcountries['CF Volatility'] = winsorize(summary_statistics_allcountries['CF Volatility'], limits=[0.01, 0.01])
    summary_statistics_allcountries['Tangibility'] = winsorize(summary_statistics_allcountries['Tangibility'], limits=[0.01, 0.01])
    summary_statistics_allcountries['Book Leverage'] = winsorize(summary_statistics_allcountries['Book Leverage'], limits=[0.01, 0.01])
    summary_statistics_allcountries['Advertising'] = winsorize(summary_statistics_allcountries['Advertising'], limits=[0.01, 0.01])
    summary_statistics_allcountries = summary_statistics_allcountries.transpose()
    mean = summary_statistics_allcountries.mean(axis=1)
    median = summary_statistics_allcountries.median(axis=1)
    ss_allcountries_table.append(f'{mean}_{country}, {median}_{country}')
    ss_allcountries_table_df[f'Mean_{country}'] = mean
    ss_allcountries_table_df[f'Median_{country}'] = median
#print(ss_allcountries_table_df.to_latex())

    summary_statistics_allcountries_table['Mean'] = summary_statistics_allcountries.mean(axis=1)
    summary_statistics_allcountries_table['Median'] = summary_statistics_allcountries.median(axis=1)
## Autocorrelation test Breusch-Godfrey
data = spec_data_needed.copy()
data.loc[:, 'HHI'] = ct.add_constant(data)
ivar = ['ln Size', 'M/B', 'Profitability_x', 'CF Volatility', 'Tangibility', 'Book Leverage', 'Advertising']
import statsmodels.regression.linear_model as rg
reg = rg.OLS(data['HHI'], data[ivar], hasconst=bool).fit()
print('test stat:', np.round(dg.acorr_breusch_godfrey(reg, nlags=20) [0], 6))
print('P-value:', np.round(dg.acorr_breusch_godfrey(reg, nlags=20) [1], 6))

## whites test
#perform White's test
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_white
data = spec_data_needed.copy()
y = data['HHI']

#define predictor variables
x = data[['ln Size', 'M/B', 'Profitability_x', 'CF Volatility', 'Tangibility', 'Book Leverage', 'Advertising']]

#add constant to predictor variables
x = sm.add_constant(x)

#fit regression model
model = sm.OLS(y, x).fit()
white_test = het_white(model.resid,  model.model.exog)

#define labels to use for output of White's test
labels = ['Test Statistic', 'Test Statistic p-value', 'F-Statistic', 'F-Test p-value']

#print results of White's test
print(dict(zip(labels, white_test)))


## statistikk tabell for naics sector
industry_statistic = combined_dataset.copy()
industry_statistic_df = pd.DataFrame()
industry_statistic_result = pd.DataFrame()

for naics in industry_statistic['NAICS Sector Code'].unique():
    if naics in exclude_naics:
        continue
    industry_statistic_naics = industry_statistic[industry_statistic['NAICS Sector Code'] == naics]
    industry_statistic_naics['Other Borrowings/Total Debt'] = industry_statistic_naics['Other Borrowings/Total Debt'] + industry_statistic_naics['Trust Preferred/Total Debt']
    industry_statistic_naics = industry_statistic_naics.drop('Trust Preferred/Total Debt', axis=1)
    industry_statistic_naics = industry_statistic_naics[['Revolving Credit/Total Debt',
                                                                        'Term Loans/Total Debt',
                                                                        'Bonds and Notes/Total Debt',
                                                                        'Commercial Paper/Total Debt',
                                                                        'Capital Lease/Total Debt',
                                                                        'Other Borrowings/Total Debt',
                                                                        'HHI']].dropna()

    naics_dict = {}
    for col in industry_statistic_naics.columns:
        percentage = industry_statistic_naics[col].mean() * 100
        naics_dict[f'{col}_percentage'] = percentage
    industry_statistic_result = industry_statistic_result.append(naics_dict, ignore_index=True)
print(industry_statistic_result.to_latex())
## same for country
country_statistic = combined_dataset.copy()
country_statistic_df = pd.DataFrame()

for country in country_statistic['Country of Exchange'].unique():
    country_statistic_naics = country_statistic[country_statistic['Country of Exchange'] == country]
    country_statistic_naics['Other Borrowings/Total Debt'] = country_statistic_naics['Other Borrowings/Total Debt'] + country_statistic_naics['Trust Preferred/Total Debt']
    country_statistic_naics = country_statistic_naics.drop('Trust Preferred/Total Debt', axis=1)
    country_statistic_naics = country_statistic_naics[['Revolving Credit/Total Debt',
                                                                        'Term Loans/Total Debt',
                                                                        'Bonds and Notes/Total Debt',
                                                                        'Commercial Paper/Total Debt',
                                                                        'Capital Lease/Total Debt',
                                                                        'Other Borrowings/Total Debt',
                                                                        'HHI']].dropna()

    country_dict = {}
    for col in country_statistic_naics.columns:
        percentage = country_statistic_naics[col].mean() * 100
        country_dict[f'{col}_percentage'] = percentage
    country_statistic_df = country_statistic_df.append(country_dict, ignore_index=True)
print(country_statistic_df.to_latex())
## statistikk for naics sector  med gjenonnomsnitt til ulike firm characteristics
industry_statistic = spec_data_combined.copy()
industry_statistic_df = pd.DataFrame()

for naics in industry_statistic['NAICS Sector Code'].unique():
    if naics in exclude_naics:
        continue
    industry_statistic_naics = industry_statistic[industry_statistic['NAICS Sector Code'] == naics]
    industry_statistic_naics = industry_statistic_naics[['ln Size', 'ln Sales', 'M/B', 'Profitability_x', 'Dividend Payer', 'CF Volatility',
                                           'Tangibility', 'Book Leverage', 'Advertising', 'HHI']].dropna()

    naics_dict = {}
    for col in industry_statistic_naics.columns:
        percentage = industry_statistic_naics[col].mean()
        naics_dict[f'{col}'] = percentage
    industry_statistic_df = industry_statistic_df.append(naics_dict, ignore_index=True)
industry_statistic_df = industry_statistic_df.transpose()
print(industry_statistic_df.to_latex())

################### regression industry specific ##############
for industry in combined_dataset['NAICS Sector Code'].unique():
    globals()[f"spec_data_{industry}"] = spec_data_combined[['ln Size', 'M/B', 'Profitability_x', 'CF Volatility',
                                              'Tangibility', 'Book Leverage', 'Advertising', 'Instrument',
                                              'Date', 'HHI', 'DS90 dummy', 'NAICS Sector Code']]
    globals()[f"spec_data_{industry}"]['Date'] = globals()[f"spec_data_{industry}"]['Date'].dt.year
    globals()[f"spec_data_{industry}"] = globals()[f"spec_data_{industry}"][globals()[f"spec_data_{industry}"]['NAICS Sector Code'] == industry]
    globals()[f"spec_data_{industry}"].set_index(['Instrument', 'Date'], inplace=True)
    shifted = globals()[f"spec_data_{industry}"].groupby(level='Instrument').shift(-1)
    globals()[f"spec_data_{industry}"].join(shifted.rename(columns=lambda x: x + '_lag'))
    globals()[f"spec_data_{industry}_merged"] = pd.merge(globals()[f"spec_data_{industry}"], shifted['HHI'], left_index=True, right_index=True, how='left')
    globals()[f"spec_data_{industry}_merged"].reset_index(inplace=True)
    globals()[f"spec_data_{industry}_merged"].drop(['HHI_x', 'DS90 dummy', 'Instrument', 'NAICS Sector Code'], axis=1, inplace=True)
    globals()[f"spec_data_{industry}_merged"].dropna(inplace=True)
    #dummies_NAICS = pd.get_dummies(globals()[f"spec_data_{industry}_merged"]['NAICS Sector Code'])
    dummies_year = pd.get_dummies(globals()[f"spec_data_{industry}_merged"]['Date'])
    globals()[f"spec_data_{industry}_merged"] = pd.concat([globals()[f"spec_data_{industry}_merged"], dummies_year], axis=1)
    globals()[f"spec_data_{industry}_merged"].drop(['Date'], axis=1, inplace=True)
    globals()[f"spec_data_{industry}_merged"].dropna(inplace=True)
    globals()[f"spec_data_{industry}_merged"]['M/B'] = winsorize(globals()[f"spec_data_{industry}_merged"]['M/B'], limits=[0.01, 0.01])
    globals()[f"spec_data_{industry}_merged"]['Profitability_x'] = winsorize(globals()[f"spec_data_{industry}_merged"]['Profitability_x'], limits=[0.001, 0.001])
    globals()[f"spec_data_{industry}_merged"]['CF Volatility'] = winsorize(globals()[f"spec_data_{industry}_merged"]['CF Volatility'], limits=[0.01, 0.01])
    globals()[f"spec_data_{industry}_merged"]['Tangibility'] = winsorize(globals()[f"spec_data_{industry}_merged"]['Tangibility'], limits=[0.001, 0.001])
    globals()[f"spec_data_{industry}_merged"]['Book Leverage'] = winsorize(globals()[f"spec_data_{industry}_merged"]['Book Leverage'], limits=[0.001, 0.001])
    #globals()[f"spec_data_{industry}_merged"]['CAPEX'] = winsorize(globals()[f"spec_data_{industry}_merged"]['CAPEX'], limits=[0.001, 0.001])
    globals()[f"spec_data_{industry}_merged"]['Advertising'] = winsorize(globals()[f"spec_data_{industry}_merged"]['Advertising'], limits=[0.001, 0.001])

    Y = globals()[f"spec_data_{industry}_merged"]['HHI_y']
    X = globals()[f"spec_data_{industry}_merged"].drop(globals()[f"spec_data_{industry}_merged"].columns[[8, 9, -1]], axis=1)
    X = sm.add_constant(X)
    globals()[f"spec_reg_{industry}"] = sm.OLS(Y, X).fit(cov_type="HC0")
    print(f"Regression for {industry}")
    print(globals()[f"spec_reg_{industry}"].summary())