import pandas as pd
import polars as pl
import matplotlib as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import ttest_ind
import scipy.stats as stats
from scipy.stats import norm
import statsmodels.api as sm

refinitivdata = pd.read_csv('refinitivdata4.csv')
assetsusd =  pd.read_csv('assetsusd.csv')
new_column_names = {'Instrument': 'Instrument', 'Total Assets': 'Total Assets USD', 'Date': 'Date'}
assetsusd = assetsusd.rename(columns=new_column_names)
assetsusd = assetsusd.drop_duplicates(subset=['Instrument', 'Date'])
refinitivdata = refinitivdata.drop_duplicates(subset=['Instrument', 'Date'])
refinitivdata = pd.merge(refinitivdata, assetsusd, on=['Instrument', 'Date'])
# fixing the dataframe
refinitivdata[['Company Common Name', 'NAICS Sector Code',
               'NAICS Subsector Code',  'NAICS National Industry Code',
                'NAICS Industry Group Code',
               'Country of Exchange',  'Market Capitalization', 'Net Income after Tax',
               "Total Shareholders' Equity incl Minority Intr & Hybrid Debt"]] = refinitivdata.groupby('Instrument')[
    ['Company Common Name', 'NAICS Sector Code',
               'NAICS Subsector Code',  'NAICS National Industry Code',
                'NAICS Industry Group Code',
               'Country of Exchange',  'Market Capitalization', 'Net Income after Tax',
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

## CLUSTER ANALYSIS ##
scatterdata = combined_dataset.copy()
scatterdata['Other Borrowings/Total Debt'] = scatterdata['Other Borrowings/Total Debt'] + scatterdata['Trust Preferred/Total Debt']

## creating a subplot of scatterdata to find patterns for the clusters
clusterpatterns = scatterdata[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                               'Other Borrowings/Total Debt', 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt']]

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
plt.pyplot.plot(frame['Cluster'], frame['SSE'], marker='o')
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

### adding Market Leverage, Liquidity and Size to  the scatterdata dataset
scatterdata['MV Equity'] = scatterdata['Price Close'] * scatterdata['Common Shares - ''Outstanding - Total - ''Ord/DR/CPO']
scatterdata['Market Leverage'] = scatterdata['Debt - Total'] / (scatterdata['Debt - Total'] + scatterdata['MV Equity'])
scatterdata['Liquidity'] = scatterdata['Total Current Assets'] / scatterdata['Total Current Liabilities']
scatterdata['Size'] = scatterdata['Total Assets USD']


#### convert to polar ####
scatterdata_polar = scatterdata[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Other Borrowings/Total Debt', 'Capital Lease/Total Debt',
                                 'Commercial Paper/Total Debt', 'clusters', 'HHI',
                                 'ROE', 'MV Equity', 'Market Leverage', 'Liquidity', 'Size']]
scatterdata_polar = pl.from_pandas(scatterdata_polar[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt',
                                 'Revolving Credit/Total Debt', 'Other Borrowings/Total Debt', 'Capital Lease/Total Debt',
                                 'Commercial Paper/Total Debt', 'clusters', 'HHI',
                                 'ROE', 'MV Equity', 'Market Leverage', 'Liquidity', 'Size']])

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
        pl.mean('ROE'),
        pl.mean('Market Leverage'),
        pl.mean('Liquidity'),
        pl.mean('Size')
    ]
).to_pandas()
scatterdata_polar.set_index('clusters', inplace=True)
scatterdata_polar.sort_index(ascending=True, inplace=True)

datafor3d = scatterdata_polar[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                               'Other Borrowings/Total Debt', 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt']]
##### prøver å lage 3d chart av clusteringen
result = np.array(datafor3d)
colors = ['r', 'b', 'g', 'y', 'b', 'p']
fig = plt.pyplot.figure(figsize=(8, 8), dpi=250)
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
plt.pyplot.show()

## Table for concentration of debt specialization
debtconcentrationdf = combined_dataset.copy()
debtconcentrationdf['Other Borrowings/Total Debt'] = debtconcentrationdf['Other Borrowings/Total Debt'] + debtconcentrationdf['Trust Preferred/Total Debt']
debtconcentrationdf = debtconcentrationdf[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
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

## Debtconcentration for each country individually
# NORWAY #
debtconcentration_norway = combined_dataset.copy()
debtconcentration_norway = debtconcentration_norway[debtconcentration_norway['Country of Exchange'] == 'Norway']
debtconcentration_norway['Other Borrowings/Total Debt'] = debtconcentration_norway['Other Borrowings/Total Debt'] + debtconcentration_norway['Trust Preferred/Total Debt']
debtconcentration_norway = debtconcentration_norway[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
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
debtconcentration_sweden['Other Borrowings/Total Debt'] = debtconcentration_sweden['Other Borrowings/Total Debt'] + debtconcentration_sweden['Trust Preferred/Total Debt']
debtconcentration_sweden = debtconcentration_sweden[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
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
debtconcentration_denmark['Other Borrowings/Total Debt'] = debtconcentration_denmark['Other Borrowings/Total Debt'] + debtconcentration_denmark['Trust Preferred/Total Debt']
debtconcentration_denmark = debtconcentration_denmark[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
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
debtconcentration_finland['Other Borrowings/Total Debt'] = debtconcentration_finland['Other Borrowings/Total Debt'] + debtconcentration_finland['Trust Preferred/Total Debt']
debtconcentration_finland = debtconcentration_finland[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
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
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Term Loans/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage = pd.DataFrame()
conditional_debt_concentration_percentage['TL50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Bonds and Notes/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['B&N50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Revolving Credit/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['RC50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Capital Lease/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['CL50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Commercial Paper/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['CP50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Other Borrowings/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['OB50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration_percentage = conditional_debt_concentration_percentage.transpose()

## conditional debt concentration
conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Term Loans/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage = pd.DataFrame()
conditional_debt_concentration_percentage['TL50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Bonds and Notes/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['B&N50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Revolving Credit/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['RC50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Capital Lease/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['CL50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Commercial Paper/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['CP50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration = combined_dataset.copy()
conditional_debt_concentration['Other Borrowings/Total Debt'] = conditional_debt_concentration['Other Borrowings/Total Debt'] + conditional_debt_concentration['Trust Preferred/Total Debt']
conditional_debt_concentration = conditional_debt_concentration[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration = conditional_debt_concentration[conditional_debt_concentration['Other Borrowings/Total Debt'] >= 0.5]
conditional_debt_concentration = conditional_debt_concentration.transpose()
conditional_debt_concentration_percentage['OB50avg'] = conditional_debt_concentration.mean(axis=1)

conditional_debt_concentration_percentage = conditional_debt_concentration_percentage.transpose()

## conditional debt concentration on industry
conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['NAICS Sector Code'].isin(['72'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry['Other Borrowings/Total Debt'] + conditional_debt_concentration_industry['Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['Term Loans/Total Debt'] >= 0.5]
conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage = pd.DataFrame()
conditional_debt_concentration_percentage['TL50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['NAICS Sector Code'].isin(['72'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry['Other Borrowings/Total Debt'] + conditional_debt_concentration_industry['Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['Bonds and Notes/Total Debt'] >= 0.5]
conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage['B&N50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['NAICS Sector Code'].isin(['72'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry['Other Borrowings/Total Debt'] + conditional_debt_concentration_industry['Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['Revolving Credit/Total Debt'] >= 0.5]
conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage['RC50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['NAICS Sector Code'].isin(['72'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry['Other Borrowings/Total Debt'] + conditional_debt_concentration_industry['Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['Capital Lease/Total Debt'] >= 0.5]
conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage['CL50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['NAICS Sector Code'].isin(['72'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry['Other Borrowings/Total Debt'] + conditional_debt_concentration_industry['Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['Commercial Paper/Total Debt'] >= 0.5]
conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage['CP50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_industry = combined_dataset.copy()
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['NAICS Sector Code'].isin(['72'])]
conditional_debt_concentration_industry['Other Borrowings/Total Debt'] = conditional_debt_concentration_industry['Other Borrowings/Total Debt'] + conditional_debt_concentration_industry['Trust Preferred/Total Debt']
conditional_debt_concentration_industry = conditional_debt_concentration_industry[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
conditional_debt_concentration_industry = conditional_debt_concentration_industry[conditional_debt_concentration_industry['Other Borrowings/Total Debt'] >= 0.5]
conditional_debt_concentration_industry = conditional_debt_concentration_industry.transpose()
conditional_debt_concentration_percentage['OB50avg'] = conditional_debt_concentration_industry.mean(axis=1)

conditional_debt_concentration_percentage = conditional_debt_concentration_percentage.transpose()

### Debt specialization (Which firms specialize)
which_firms_specialize = pd.read_csv('which_firms_specialize.csv')
which_firms_specialize.fillna(0, inplace=True)
which_firms_specialize.rename(columns={'Debt - Total': 'Debt - Total USD', 'Market Capitalization': 'Market Cap USD', 'Price Close': 'Price Close USD'}, inplace=True)
which_firms_specialize = which_firms_specialize.drop_duplicates(subset=['Instrument', 'Date'])
which_firms_specialize['Date'] = pd.to_datetime(which_firms_specialize['Date'], format='%Y/%m/%d')
which_firms_specialize = pd.merge(refinitivdata, which_firms_specialize, on=['Instrument', 'Date'])
which_firms_specialize = which_firms_specialize.fillna(0)
which_firms_specialize['Revenue with constant'] = which_firms_specialize['Revenue from Business Activities - Total'] + 1
which_firms_specialize['Revenue with constant'] = which_firms_specialize['Revenue with constant'][which_firms_specialize['Revenue with constant'] > 0]
which_firms_specialize['Assets with constant'] = which_firms_specialize['Total Assets USD'] + 1
which_firms_specialize['Assets with constant'] = which_firms_specialize['Assets with constant'][which_firms_specialize['Assets with constant'] > 0]

which_firms_specialize['Market Cap USD'] = which_firms_specialize['Market Cap USD'].drop_duplicates()
which_firms_specialize['Net Income after Tax'] = which_firms_specialize['Net Income after Tax'].drop_duplicates()
which_firms_specialize["Total Shareholders' Equity incl Minority Intr & Hybrid Debt"] = which_firms_specialize["Total Shareholders' Equity incl Minority Intr & Hybrid Debt"].drop_duplicates()
which_firms_specialize['Date'] = pd.to_datetime(which_firms_specialize['Date'])
which_firms_specialize = which_firms_specialize[which_firms_specialize['Date'].dt.year.isin(range(2001, 2022))]

spec_data_needed = pd.DataFrame()
spec_data_needed['ln Size'] = np.log(which_firms_specialize['Assets with constant'])
spec_data_needed['ln Sales'] = np.log(which_firms_specialize['Revenue with constant'])
spec_data_needed['M/B'] = which_firms_specialize['Market Cap USD'] / which_firms_specialize['Total Assets USD']
spec_data_needed['ROE'] = which_firms_specialize['ROE']
spec_data_needed['Dividend Payer'] = which_firms_specialize['Dividend Per Share - Mean'].apply(lambda x: 1 if x != 0 else 0)
spec_data_needed['Cash Holdings'] = which_firms_specialize['Cash & Short Term Investments'] / which_firms_specialize['Total Assets USD']
spec_data_needed['Tangibility'] = which_firms_specialize['PPE - Net Percentage of Total Assets']
spec_data_needed['Book Leverage'] = which_firms_specialize['Debt - Total USD'] / which_firms_specialize['Total Assets USD']
spec_data_needed['CAPEX'] = which_firms_specialize['CAPEX Percentage of Total Assets']
spec_data_needed['CAPEX'] = spec_data_needed['CAPEX']/100
spec_data_needed['Advertising'] = which_firms_specialize['Selling General & Administrative Expenses - Total'] / which_firms_specialize['Total Assets USD']
spec_data_needed['Instrument'] = which_firms_specialize['Instrument']
spec_data_needed['Date'] = which_firms_specialize['Date']
spec_data_needed = spec_data_needed.replace([np.inf, -np.inf], np.nan).dropna()

# merging spec_data_needed with HHI and DS90
spec_data_combined = pd.merge(spec_data_needed, combined_dataset, on=['Instrument', 'Date'])
spec_data_combined.dropna(axis=0, inplace=True)

spec_data_needed = spec_data_combined[['ln Size', 'ln Sales', 'M/B', 'ROE_x', 'Dividend Payer', 'Cash Holdings',
                                       'Tangibility', 'Book Leverage', 'CAPEX', 'Advertising', 'Instrument', 'Date',
                                       'HHI', 'DS90 dummy']]
# Correlation between variables
rho = spec_data_needed.corr()
pval = spec_data_needed.corr(method=lambda x, y: pearsonr(x, y) [1]) - np.eye(*rho.shape)
p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
rho.round(2).astype(str) + p


spec_data_needed_sorted = spec_data_needed.sort_values('HHI')
spec_data_needed_sorted['quartile'] = pd.qcut(spec_data_needed_sorted['HHI'], q=3, labels=['1st tertile', '2nd tertile', '3rd tertile'])


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
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales','DS90 dummy'], axis =1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date']], how='left', on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x+'_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)


Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg = sm.OLS(Y, X)
multivariate_reg_res = multivariate_reg.fit(cov_type = "HC0")
print(multivariate_reg_res.summary())
### adding NAICS dummies
multivariate_reg1 = spec_data_needed.copy()
multivariate_reg1 = multivariate_reg1.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales','DS90 dummy'], axis =1)
multivariate_reg1 = pd.merge(multivariate_reg1, which_firms_specialize[['Instrument', 'Date', 'NAICS Sector Code']], how='left', on=['Instrument', 'Date'])
multivariate_reg1['Date'] = multivariate_reg1['Date'].dt.year

dummies_NAICS = pd.get_dummies(multivariate_reg1['NAICS Sector Code'])
multivariate_merge1 = pd.concat([multivariate_reg1, dummies_NAICS], axis=1)
multivariate_merge1.drop(['NAICS Sector Code', 'Date', 'Instrument'], axis=1, inplace=True)

Y = multivariate_merge1['HHI'].fillna(0)
X = multivariate_merge1.drop(['HHI', '62'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg1 = sm.OLS(Y, X)
multivariate_reg_res1 = multivariate_reg1.fit(cov_type = "HC0")
print(multivariate_reg_res1.summary())

### only country dummy
multivariate_reg2 = spec_data_needed.copy()
multivariate_reg2 = multivariate_reg2.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales','DS90 dummy'], axis =1)
multivariate_reg2 = pd.merge(multivariate_reg2, which_firms_specialize[['Instrument', 'Date', 'Country of Exchange']], how='left', on=['Instrument', 'Date'])
multivariate_reg2['Date'] = multivariate_reg2['Date'].dt.year

dummies_country = pd.get_dummies(multivariate_reg2['Country of Exchange'])
multivariate_merge2 = pd.concat([multivariate_reg2, dummies_country], axis=1)
multivariate_merge2.drop(['Country of Exchange', 'Date', 'Instrument'], axis=1, inplace=True)

Y = multivariate_merge2['HHI'].fillna(0)
X = multivariate_merge2.drop(['HHI', 'Denmark'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg2 = sm.OLS(Y, X)
multivariate_reg_res2 = multivariate_reg2.fit(cov_type = "HC0")
print(multivariate_reg_res2.summary())

### year dummies only
multivariate_reg3 = spec_data_needed.copy()
multivariate_reg3 = multivariate_reg3.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales','DS90 dummy'], axis =1)
multivariate_reg3 = pd.merge(multivariate_reg3, which_firms_specialize[['Instrument', 'Date']], how='left', on=['Instrument', 'Date'])
multivariate_reg3['Date'] = multivariate_reg3['Date'].dt.year

dummies_year = pd.get_dummies(multivariate_reg3['Date'])
multivariate_merge3 = pd.concat([multivariate_reg3, dummies_year], axis=1)
multivariate_merge3.drop(['Date', 'Instrument'], axis=1, inplace=True)

Y = multivariate_merge3['HHI'].fillna(0)
X = multivariate_merge3.drop(['HHI', 2001], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg3 = sm.OLS(Y, X)
multivariate_reg_res3 = multivariate_reg3.fit(cov_type = "HC0")
print(multivariate_reg_res3.summary())


##

# count unique observations and their percentages
counts = combined_dataset['NAICS Sector Code'].value_counts()
percentages = counts / len(combined_dataset) * 100

# print the results
for idx, val in enumerate(counts.index):
    count = counts.get(val, default=0)
    percentage = count / len(combined_dataset) * 100
    print(f"{val}: {count} ({percentage:.2f}%)")

##
#debtconcentrationdf = combined_dataset.copy()
#debtconcentrationdf = debtconcentrationdf[debtconcentrationdf['NAICS Sector Code'].isin(['21'])]
#debtconcentrationdf['Other Borrowings/Total Debt'] = debtconcentrationdf['Other Borrowings/Total Debt'] + debtconcentrationdf['Trust Preferred/Total Debt']
#debtconcentrationdf = debtconcentrationdf[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
#                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']]
#
#
# thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
# percentages_df = pd.DataFrame()
#
#
#for threshold in thresholds:
#    threshold_dict = {}
#    for col in debtconcentrationdf.columns:
#        percentage = (debtconcentrationdf[col] >= threshold).mean() * 100
#        threshold_dict[f'{col}_percentage'] = percentage
#    percentages_df = percentages_df.append(threshold_dict, ignore_index=True)
#
#percentages_df = percentages_df.transpose(#)