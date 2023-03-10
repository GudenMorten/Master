import pandas as pd
import seaborn as sns
import polars as pl
import matplotlib as plt
import plotly.graph_objects as go
import matplotlib.ticker as mtick

### PLOT 1 ###

df_plot1_debt_total = combined_dataset[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']]
df_plot1_debt_total_polar = pl.from_pandas(
    df_plot1_debt_total[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']])

df_plot1_debt_total_polar = df_plot1_debt_total_polar.groupby(
    [
        "year"
    ]
).agg(
    [
        pl.sum("Short-Term Debt & Current Portion of Long-Term Debt").alias("short-term debt total"),
        pl.sum('Debt - Long-Term - Total').alias('long-term debt total'),
    ]
).to_pandas()

chart_debt_total = df_plot1_debt_total_polar.set_index('year').sort_index(ascending=True).plot(kind='bar', stacked=True)
chart_debt_total.set_xticklabels(chart_debt_total.get_xticklabels(), rotation='vertical')

# Show the plot
plt.pyplot.show()

#### PLOT 2 ####

df_plot2_debt_totalpct = combined_dataset[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Debt - Total']]
df_plot2_debt_totalpct_polar = pl.from_pandas(
    df_plot2_debt_totalpct[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Debt - Total']])

df_plot2_debt_totalpct_polar = df_plot2_debt_totalpct_polar.groupby(
    [
        "year"
    ]
).agg(
    [
        pl.sum("Short-Term Debt & Current Portion of Long-Term Debt").alias("short-term debt total"),
        pl.sum('Debt - Long-Term - Total').alias('long-term debt total'),
        pl.sum('Debt - Total').alias('debt - total')
    ]
).to_pandas()

df_plot2_debt_totalpct_polar['short term pct'] = df_plot2_debt_totalpct_polar['short-term debt total']/df_plot2_debt_totalpct_polar['debt - total']
df_plot2_debt_totalpct_polar['long term pct'] = df_plot2_debt_totalpct_polar['long-term debt total']/df_plot2_debt_totalpct_polar['debt - total']
df_plot2_debt_totalpct_polar = df_plot2_debt_totalpct_polar[['short term pct', 'long term pct', 'year']]
chart_debt_totalpct = df_plot2_debt_totalpct_polar.set_index('year').sort_index(ascending=True).plot(kind='bar', stacked=True)
chart_debt_totalpct.set_xticklabels(chart_debt_totalpct.get_xticklabels(), rotation='vertical')

# Show the plot
plt.pyplot.show()


### PLOT 3 FINLAND ###
df_plot3finland_debt_total = combined_dataset[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Country of Exchange']]
df_plot3finland_debt_total = combined_dataset.loc[combined_dataset['Country of Exchange'] == 'Finland', ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']]
df_plot3finland_debt_total_polar = pl.from_pandas(
    df_plot3finland_debt_total[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']])

df_plot3finland_debt_total_polar = df_plot3finland_debt_total_polar.groupby(
    [
        'year'
    ]
).agg(
    [
        pl.sum("Short-Term Debt & Current Portion of Long-Term Debt").alias("short-term debt total"),
        pl.sum('Debt - Long-Term - Total').alias('long-term debt total'),
    ]
).to_pandas()

chart_debt3finland_total = df_plot3_debt_total_polar.set_index('year').sort_index(ascending=True).plot(kind='bar', stacked=True)
chart_debt3finland_total.set_xticklabels(chart_debt3finland_total.get_xticklabels(), rotation='vertical')

# Show the plot
plt.pyplot.show()

### PLOT 3 NORWAY ###
df_plot3norway_debt_total = combined_dataset[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Country of Exchange']]
df_plot3norway_debt_total = combined_dataset.loc[combined_dataset['Country of Exchange'] == 'Norway', ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']]
df_plot3norway_debt_total_polar = pl.from_pandas(
    df_plot3norway_debt_total[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']])

df_plot3norway_debt_total_polar = df_plot3norway_debt_total_polar.groupby(
    [
        'year'
    ]
).agg(
    [
        pl.sum("Short-Term Debt & Current Portion of Long-Term Debt").alias("short-term debt total"),
        pl.sum('Debt - Long-Term - Total').alias('long-term debt total'),
    ]
).to_pandas()

chart_debt3norway_total = df_plot3_debt_total_polar.set_index('year').sort_index(ascending=True).plot(kind='bar', stacked=True)
chart_debt3norway_total.set_xticklabels(chart_debt3norway_total.get_xticklabels(), rotation='vertical')

# Show the plot
plt.pyplot.show()
### PLOT 3 SWEDEN ###
df_plot3sweden_debt_total = combined_dataset[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Country of Exchange']]
df_plot3sweden_debt_total = combined_dataset.loc[combined_dataset['Country of Exchange'] == 'Sweden', ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']]
df_plot3sweden_debt_total_polar = pl.from_pandas(
    df_plot3sweden_debt_total[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']])

df_plot3sweden_debt_total_polar = df_plot3sweden_debt_total_polar.groupby(
    [
        'year'
    ]
).agg(
    [
        pl.sum("Short-Term Debt & Current Portion of Long-Term Debt").alias("short-term debt total"),
        pl.sum('Debt - Long-Term - Total').alias('long-term debt total'),
    ]
).to_pandas()

chart_debt3sweden_total = df_plot3_debt_total_polar.set_index('year').sort_index(ascending=True).plot(kind='bar', stacked=True)
chart_debt3sweden_total.set_xticklabels(chart_debt3sweden_total.get_xticklabels(), rotation='vertical')

# Show the plot
plt.pyplot.show()
### PLOT 3 DENMARK ###
df_plot3denmark_debt_total = combined_dataset[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Country of Exchange']]
df_plot3denmark_debt_total = combined_dataset.loc[combined_dataset['Country of Exchange'] == 'Denmark', ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']]
df_plot3denmark_debt_total_polar = pl.from_pandas(
    df_plot3denmark_debt_total[['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']])

df_plot3denmark_debt_total_polar = df_plot3denmark_debt_total_polar.groupby(
    [
        'year'
    ]
).agg(
    [
        pl.sum("Short-Term Debt & Current Portion of Long-Term Debt").alias("short-term debt total"),
        pl.sum('Debt - Long-Term - Total').alias('long-term debt total'),
    ]
).to_pandas()

chart_debt3denmark_total = df_plot3_debt_total_polar.set_index('year').sort_index(ascending=True).plot(kind='bar', stacked=True)
chart_debt3denmark_total.set_xticklabels(chart_debt3denmark_total.get_xticklabels(), rotation='vertical')

# Show the plot
plt.pyplot.show()

### PLOT 7 ###
plot7_line_totaldebt = combined_dataset[['year', 'Country of Exchange', 'Debt - Total']]
plot7_line_totaldebt_polar = pl.from_pandas(
    plot7_line_totaldebt[['year', 'Country of Exchange', 'Debt - Total']])

plot7_line_totaldebt_polar = plot7_line_totaldebt_polar.groupby(
    [
        'year', 'Country of Exchange'
    ]
).agg(
    [
        pl.sum('Debt - Total').alias('total debt')
    ]
).to_pandas()

plot7_line_totaldebt = plot7_line_totaldebt_polar.set_index('year').sort_index(ascending=True)
sns.lineplot(data=plot7_line_totaldebt, x=plot7_line_totaldebt.index, y='total debt', hue='Country of Exchange')

plt.pyplot.show()

### PLOT 8 ###
plot8_line_longterm = combined_dataset[['year', 'Country of Exchange', 'Debt - Long-Term - Total', 'Debt - Total']]
plot8_line_longterm_polar = pl.from_pandas(
    plot8_line_longterm[['year', 'Country of Exchange', 'Debt - Long-Term - Total', 'Debt - Total']])

plot8_line_longterm_polar = plot8_line_longterm_polar.groupby(
    [
        'year', 'Country of Exchange'
    ]
).agg(
    [
        pl.sum('Debt - Long-Term - Total').alias('Debt - Long-Term - Total'),
        pl.sum('Debt - Total').alias('total debt')
    ]
).to_pandas()

plot8_line_longterm = plot8_line_longterm_polar.set_index('year').sort_index(ascending=True)
plot8_line_longterm['long term pct'] = plot8_line_longterm['Debt - Long-Term - Total']/plot8_line_longterm['total debt']
plot8_line_longterm = plot8_line_longterm[['long term pct', 'Country of Exchange']]
sns.lineplot(data=plot8_line_longterm, x=plot8_line_longterm.index, y='long term pct', hue='Country of Exchange')

plt.pyplot.show()

### PLOT 9 ###
plot9_line_shortterm = combined_dataset[['year', 'Country of Exchange', 'Short-Term Debt & Current Portion of Long-Term Debt', 'Debt - Total']]
plot9_line_shortterm_polar = pl.from_pandas(
    plot9_line_shortterm[['year', 'Country of Exchange', 'Short-Term Debt & Current Portion of Long-Term Debt', 'Debt - Total']])

plot9_line_shortterm_polar = plot9_line_shortterm_polar.groupby(
    [
        'year', 'Country of Exchange'
    ]
).agg(
    [
        pl.sum('Short-Term Debt & Current Portion of Long-Term Debt').alias('Short-Term Debt & Current Portion of Long-Term Debt'),
        pl.sum('Debt - Total').alias('total debt')
    ]
).to_pandas()

plot9_line_shortterm = plot9_line_shortterm_polar.set_index('year').sort_index(ascending=True)
plot9_line_shortterm['short term debt pct'] = plot9_line_shortterm['Short-Term Debt & Current Portion of Long-Term Debt']/plot9_line_shortterm['total debt']
plot9_line_shortterm = plot9_line_shortterm[['short term debt pct', 'Country of Exchange']].fillna(0)
sns.lineplot(data=plot9_line_shortterm, x=plot9_line_shortterm.index, y='short term debt pct', hue='Country of Exchange')

plt.pyplot.show()


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