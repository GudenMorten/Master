import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

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
                         (combined_dataset['Commercial Paper'] / combined_dataset['TD']) ** 2 + (
                                 combined_dataset['Trust Preferred'] / combined_dataset['TD']) ** 2 + \
                         (combined_dataset['Other Borrowings'] / combined_dataset['TD']) ** 2
combined_dataset['HHI'] = (combined_dataset['SS'] - (1 / 7)) / (1 - (1 / 7))

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

debtconcentrationdf = pl.from_pandas(debtconcentrationdf[['Term Loans/Total Debt', 'Bonds and Notes/Total Debt', 'Revolving Credit/Total Debt',
                                 'Capital Lease/Total Debt', 'Commercial Paper/Total Debt', 'Other Borrowings/Total Debt']])

debtconcentrationdf = debtconcentrationdf.groupby(
    [
        'Term Loans/Total Debt',
        'Bonds and Notes/Total Debt',
        'Revolving Credit/Total Debt',
        'Capital Lease/Total Debt',
        'Commercial Paper/Total Debt',
        'Other Borrowings/Total Debt'
    ]
#).agg(
#    [
#        pl.mean('Term Loans/Total Debt'),
#        pl.mean('Bonds and Notes/Total Debt'),
#        pl.mean('Revolving Credit/Total Debt'),
#        pl.mean('Other Borrowings/Total Debt'),
#        pl.mean('Capital Lease/Total Debt'),
#        pl.mean('Commercial Paper/Total Debt'),
#        pl.mean('HHI'),
#        pl.mean('ROE'),
#        pl.mean('Market Leverage'),
#        pl.mean('Liquidity'),
#        pl.mean('Size')
#    ]
#).to_pandas()
#