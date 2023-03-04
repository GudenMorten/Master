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

# Merging the gvisin dataset with the refinitivdata dataset to include gvkey as well as ISIN number
refinitivdata_withgvkey= pd.merge(refinitivdata, gvisin, on='Instrument', how='left')







#chart = sns.barplot(x='Fiscal Year', y='Long-Term Debt/Total Debt', hue= 'Country of Exchange', data=refinitivdata)
#chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
#plt.pyplot.show()

#refinitivdata.dropna(subset=['Fiscal Year'], inplace=True)

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




#### PLOT FOR BONDS ###
df_plot_bonds_annual = combined_dataset[['Bonds and Notes', 'Fiscal Year', 'Country of Exchange']]
df_plot_bonds_annual = combined_dataset.loc[combined_dataset['Country of Exchange'] == 'Finland', ['Bonds and Notes', 'Fiscal Year']]
df_plot_bonds_annual_polar = pl.from_pandas(
    df_plot_bonds_annual[["Bonds and Notes", "Fiscal Year"]])

df_plot_bonds_annual_polar = df_plot_bonds_annual_polar.groupby(
    [
        "Fiscal Year"
    ]
).agg(
    [
        pl.sum("Bonds and Notes").alias("Bond Debt Total")
    ]
).to_pandas()

chart_bonds_on_country = sns.barplot(data=df_plot_bonds_annual_polar, x='Fiscal Year', y='Bond Debt Total')
chart_bonds_on_country.set_xticklabels(chart_bonds_on_country.get_xticklabels(), rotation=45)
chart_bonds_on_country.set_ylim(0,400000000)
plt.pyplot.show()

#### PLOT FOR LOANS ###
df_plot_loans_annual = combined_dataset[['Term Loans', 'Fiscal Year', 'Country of Exchange']]
df_plot_loans_annual = combined_dataset.loc[combined_dataset['Country of Exchange'] == 'Denmark', ['Term Loans', 'Fiscal Year']]
df_plot_loans_annual_polar = pl.from_pandas(
    df_plot_loans_annual[["Term Loans", "Fiscal Year"]])

df_plot_loans_annual_polar = df_plot_loans_annual_polar.groupby(
    [
        "Fiscal Year"
    ]
).agg(
    [
        pl.sum("Term Loans").alias("Term Loans Total")
    ]
).to_pandas()

chart_loans_on_country = sns.barplot(data=df_plot_loans_annual_polar, x='Fiscal Year', y='Term Loans Total')
chart_loans_on_country.set_xticklabels(chart_loans_on_country.get_xticklabels(), rotation=45)
chart_loans_on_country #.set_ylim(0,400000000)
plt.pyplot.show()
#chart_bonds_4 = sns.lineplot(data=combined_dataset, x='Fiscal Year', y='Bonds and Notes')
#plt.pyplot.show()

#### PLOT FOR TOTAL DEBT ####
df_plot_totaldebt_annual = combined_dataset[['Term Loans','Bonds and Notes','Revolving Credit','Other Borrowings', 'Capital Lease', 'Commercial Paper', 'Trust Preferred', 'Fiscal Year', 'Country of Exchange']]
df_plot_totaldebt_annual = combined_dataset.loc[combined_dataset['Country of Exchange'] == 'Denmark', ['Term Loans', 'Bonds and Notes', 'Revolving Credit', 'Other Borrowings', 'Capital Lease', 'Commercial Paper', 'Trust Preferred', 'Fiscal Year']]
df_plot_totaldebt_annual_polar = pl.from_pandas(
    df_plot_totaldebt_annual[['Term Loans','Bonds and Notes','Revolving Credit','Other Borrowings', 'Capital Lease', 'Commercial Paper', 'Trust Preferred', "Fiscal Year"]])

df_plot_totaldebt_annual_polar = df_plot_totaldebt_annual_polar.groupby(
    [
        "Fiscal Year"
    ]
).agg(
    [
        pl.sum("Term Loans").alias("Term Loans Total"),
        pl.sum("Bonds and Notes").alias("Bonds and Notes Total"),
        pl.sum("Revolving Credit").alias("Revolving Credit Total"),
        pl.sum("Other Borrowings").alias("Other Borrowings Total"),
        pl.sum("Capital Lease").alias("Capital Lease Total"),
        pl.sum("Commercial Paper").alias("Commercial Paper Total"),
        pl.sum("Trust Preferred").alias("Trust Preferred Total")
    ]
).to_pandas()

df_plot_totaldebt_annual_polar = df_plot_totaldebt_annual_polar.set_index('Fiscal Year')
chart_totaldebt_on_country = sns.lineplot(data=df_plot_totaldebt_annual_polar)
chart_totaldebt_on_country.set_xticklabels(chart_totaldebt_on_country.get_xticklabels(), rotation=45)
chart_totaldebt_on_country #.set_ylim(0,400000000)
plt.pyplot.show()



##### TEST PLOT UTEN TERM LOANS###
df_plot_totaldebt_annual = combined_dataset[['Bonds and Notes','Revolving Credit','Other Borrowings', 'Capital Lease', 'Commercial Paper', 'Trust Preferred', 'Fiscal Year', 'Country of Exchange']]
df_plot_totaldebt_annual = combined_dataset.loc[combined_dataset['Country of Exchange'] == 'Denmark', ['Bonds and Notes', 'Revolving Credit', 'Other Borrowings', 'Capital Lease', 'Commercial Paper', 'Trust Preferred', 'Fiscal Year']]
df_plot_totaldebt_annual_polar = pl.from_pandas(
    df_plot_totaldebt_annual[['Bonds and Notes','Revolving Credit','Other Borrowings', 'Capital Lease', 'Commercial Paper', 'Trust Preferred', "Fiscal Year"]])

df_plot_totaldebt_annual_polar = df_plot_totaldebt_annual_polar.groupby(
    [
        "Fiscal Year"
    ]
).agg(
    [
        pl.sum("Bonds and Notes").alias("Bonds and Notes Total"),
        pl.sum("Revolving Credit").alias("Revolving Credit Total"),
        pl.sum("Other Borrowings").alias("Other Borrowings Total"),
        pl.sum("Capital Lease").alias("Capital Lease Total"),
        pl.sum("Commercial Paper").alias("Commercial Paper Total"),
        pl.sum("Trust Preferred").alias("Trust Preferred Total")
    ]
).to_pandas()

df_plot_totaldebt_annual_polar = df_plot_totaldebt_annual_polar.set_index('Fiscal Year')
chart_totaldebt_on_country = sns.lineplot(data=df_plot_totaldebt_annual_polar)
chart_totaldebt_on_country.set_xticklabels(chart_totaldebt_on_country.get_xticklabels(), rotation=45)
chart_totaldebt_on_country #.set_ylim(0,400000000)
plt.pyplot.show()