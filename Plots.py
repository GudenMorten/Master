import pandas as pd
import seaborn as sns
import polars as pl
import matplotlib as plt
import plotly.graph_objects as go
import matplotlib.ticker as mtick

### PLOT 1 ###

df_plot1_debt_total = combined_dataset[
    ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']]
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

df_plot2_debt_totalpct = combined_dataset[
    ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Debt - Total']]
df_plot2_debt_totalpct_polar = pl.from_pandas(
    df_plot2_debt_totalpct[
        ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Debt - Total']])

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

df_plot2_debt_totalpct_polar['short term pct'] = df_plot2_debt_totalpct_polar['short-term debt total'] / \
                                                 df_plot2_debt_totalpct_polar['debt - total']
df_plot2_debt_totalpct_polar['long term pct'] = df_plot2_debt_totalpct_polar['long-term debt total'] / \
                                                df_plot2_debt_totalpct_polar['debt - total']
df_plot2_debt_totalpct_polar = df_plot2_debt_totalpct_polar[['short term pct', 'long term pct', 'year']]
chart_debt_totalpct = df_plot2_debt_totalpct_polar.set_index('year').sort_index(ascending=True).plot(kind='bar',
                                                                                                     stacked=True)
chart_debt_totalpct.set_xticklabels(chart_debt_totalpct.get_xticklabels(), rotation='vertical')

# Show the plot
plt.pyplot.show()

### PLOT 3 NORWAY ###
df_plot3norway_debt_total = combined_dataset[
    ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Country of Exchange']]
df_plot3norway_debt_total = combined_dataset.loc[
    combined_dataset['Country of Exchange'] == 'Norway', ['Debt - Long-Term - Total',
                                                          'Short-Term Debt & Current Portion of Long-Term Debt',
                                                          'year']]
df_plot3norway_debt_total_polar = pl.from_pandas(
    df_plot3norway_debt_total[
        ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']])

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

chart_debt3norway_total = df_plot3norway_debt_total_polar.set_index('year').sort_index(ascending=True).plot(kind='bar',
                                                                                                            stacked=True)
chart_debt3norway_total.set_xticklabels(chart_debt3norway_total.get_xticklabels(), rotation='vertical')

# Show the plot
plt.pyplot.show()
### PLOT 4 SWEDEN ###
df_plot4sweden_debt_total = combined_dataset[
    ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Country of Exchange']]
df_plot4sweden_debt_total = combined_dataset.loc[
    combined_dataset['Country of Exchange'] == 'Sweden', ['Debt - Long-Term - Total',
                                                          'Short-Term Debt & Current Portion of Long-Term Debt',
                                                          'year']]
df_plot4sweden_debt_total_polar = pl.from_pandas(
    df_plot4sweden_debt_total[
        ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']])

df_plot4sweden_debt_total_polar = df_plot4sweden_debt_total_polar.groupby(
    [
        'year'
    ]
).agg(
    [
        pl.sum("Short-Term Debt & Current Portion of Long-Term Debt").alias("short-term debt total"),
        pl.sum('Debt - Long-Term - Total').alias('long-term debt total'),
    ]
).to_pandas()

chart_debt4sweden_total = df_plot4sweden_debt_total_polar.set_index('year').sort_index(ascending=True).plot(kind='bar',
                                                                                                            stacked=True)
chart_debt4sweden_total.set_xticklabels(chart_debt4sweden_total.get_xticklabels(), rotation='vertical')

# Show the plot
plt.pyplot.show()
### PLOT 5 DENMARK ###
df_plot5denmark_debt_total = combined_dataset[
    ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Country of Exchange']]
df_plot5denmark_debt_total = combined_dataset.loc[
    combined_dataset['Country of Exchange'] == 'Denmark', ['Debt - Long-Term - Total',
                                                           'Short-Term Debt & Current Portion of Long-Term Debt',
                                                           'year']]
df_plot5denmark_debt_total_polar = pl.from_pandas(
    df_plot5denmark_debt_total[
        ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']])

df_plot5denmark_debt_total_polar = df_plot5denmark_debt_total_polar.groupby(
    [
        'year'
    ]
).agg(
    [
        pl.sum("Short-Term Debt & Current Portion of Long-Term Debt").alias("short-term debt total"),
        pl.sum('Debt - Long-Term - Total').alias('long-term debt total'),
    ]
).to_pandas()

chart_debt5denmark_total = df_plot5denmark_debt_total_polar.set_index('year').sort_index(ascending=True).plot(
    kind='bar', stacked=True)
chart_debt5denmark_total.set_xticklabels(chart_debt5denmark_total.get_xticklabels(), rotation='vertical')

# Show the plot
plt.pyplot.show()

### PLOT 6 FINLAND ###
df_plot6finland_debt_total = combined_dataset[
    ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year', 'Country of Exchange']]
df_plot6finland_debt_total = combined_dataset.loc[
    combined_dataset['Country of Exchange'] == 'Finland', ['Debt - Long-Term - Total',
                                                           'Short-Term Debt & Current Portion of Long-Term Debt',
                                                           'year']]
df_plot6finland_debt_total_polar = pl.from_pandas(
    df_plot6finland_debt_total[
        ['Debt - Long-Term - Total', 'Short-Term Debt & Current Portion of Long-Term Debt', 'year']])

df_plot6finland_debt_total_polar = df_plot6finland_debt_total_polar.groupby(
    [
        'year'
    ]
).agg(
    [
        pl.sum("Short-Term Debt & Current Portion of Long-Term Debt").alias("short-term debt total"),
        pl.sum('Debt - Long-Term - Total').alias('long-term debt total'),
    ]
).to_pandas()

chart_debt6finland_total = df_plot6finland_debt_total_polar.set_index('year').sort_index(ascending=True).plot(
    kind='bar', stacked=True)
chart_debt6finland_total.set_xticklabels(chart_debt6finland_total.get_xticklabels(), rotation='vertical')

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
plot8_line_longterm['long term pct'] = plot8_line_longterm['Debt - Long-Term - Total'] / plot8_line_longterm[
    'total debt']
plot8_line_longterm = plot8_line_longterm[['long term pct', 'Country of Exchange']]
sns.lineplot(data=plot8_line_longterm, x=plot8_line_longterm.index, y='long term pct', hue='Country of Exchange')

plt.pyplot.show()

### PLOT 9 ###
plot9_line_shortterm = combined_dataset[
    ['year', 'Country of Exchange', 'Short-Term Debt & Current Portion of Long-Term Debt', 'Debt - Total']]
plot9_line_shortterm_polar = pl.from_pandas(
    plot9_line_shortterm[
        ['year', 'Country of Exchange', 'Short-Term Debt & Current Portion of Long-Term Debt', 'Debt - Total']])

plot9_line_shortterm_polar = plot9_line_shortterm_polar.groupby(
    [
        'year', 'Country of Exchange'
    ]
).agg(
    [
        pl.sum('Short-Term Debt & Current Portion of Long-Term Debt').alias(
            'Short-Term Debt & Current Portion of Long-Term Debt'),
        pl.sum('Debt - Total').alias('total debt')
    ]
).to_pandas()

plot9_line_shortterm = plot9_line_shortterm_polar.set_index('year').sort_index(ascending=True)
plot9_line_shortterm['short term debt pct'] = plot9_line_shortterm[
                                                  'Short-Term Debt & Current Portion of Long-Term Debt'] / \
                                              plot9_line_shortterm['total debt']
plot9_line_shortterm = plot9_line_shortterm[['short term debt pct', 'Country of Exchange']].fillna(0)
sns.lineplot(data=plot9_line_shortterm, x=plot9_line_shortterm.index, y='short term debt pct',
             hue='Country of Exchange')

plt.pyplot.show()

### PLOT 10 ALL COUNTRIES ###

df_plot10 = combined_dataset[
    ['Fiscal Year', 'Bonds and Notes', "Term Loans", "Revolving Credit", "Other Borrowings", "Capital Lease",
     "Commercial Paper", "Trust Preferred"]]
df_plot10_polar = pl.from_pandas(
    df_plot10[
        ['Term Loans', 'Bonds and Notes', 'Revolving Credit', 'Other Borrowings', 'Capital Lease', 'Commercial Paper',
         'Trust Preferred', "Fiscal Year"]])

df_plot10_polar = df_plot10_polar.groupby(
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

df_plot10_polar = df_plot10_polar.set_index('Fiscal Year')
chart_totaldebt_on_country = sns.lineplot(data=df_plot10_polar)
chart_totaldebt_on_country.set_xticklabels(chart_totaldebt_on_country.get_xticklabels(), rotation=45)
chart_totaldebt_on_country  # .set_ylim(0,400000000)
plt.pyplot.show()

### PLOT 11 NORWAY ###

df_plot11 = combined_dataset[
    ['Fiscal Year', 'Bonds and Notes', "Term Loans", "Revolving Credit", "Other Borrowings", "Capital Lease",
     "Commercial Paper", "Trust Preferred", "Country of Exchange"]]
df_plot11 = combined_dataset.loc[
    combined_dataset['Country of Exchange'] == 'Norway', ['Fiscal Year', 'Bonds and Notes', "Term Loans",
                                                          "Revolving Credit", "Other Borrowings", "Capital Lease",
                                                          "Commercial Paper", "Trust Preferred"]]
df_plot11_polar = pl.from_pandas(
    df_plot11[
        ['Term Loans', 'Bonds and Notes', 'Revolving Credit', 'Other Borrowings', 'Capital Lease', 'Commercial Paper',
         'Trust Preferred', "Fiscal Year"]])

df_plot11_polar = df_plot11_polar.groupby(
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

df_plot11_polar = df_plot11_polar.set_index('Fiscal Year')
chart_totaldebt_on_country = sns.lineplot(data=df_plot11_polar)
chart_totaldebt_on_country.set_xticklabels(chart_totaldebt_on_country.get_xticklabels(), rotation=45)
chart_totaldebt_on_country  # .set_ylim(0,400000000)
plt.pyplot.show()

## PLOT 12 SWEDEN ###

df_plot12 = combined_dataset[
    ['Fiscal Year', 'Bonds and Notes', "Term Loans", "Revolving Credit", "Other Borrowings", "Capital Lease",
     "Commercial Paper", "Trust Preferred", "Country of Exchange"]]
df_plot12 = combined_dataset.loc[
    combined_dataset['Country of Exchange'] == 'Sweden', ['Fiscal Year', 'Bonds and Notes', "Term Loans",
                                                          "Revolving Credit", "Other Borrowings", "Capital Lease",
                                                          "Commercial Paper", "Trust Preferred"]]
df_plot12_polar = pl.from_pandas(
    df_plot12[
        ['Term Loans', 'Bonds and Notes', 'Revolving Credit', 'Other Borrowings', 'Capital Lease', 'Commercial Paper',
         'Trust Preferred', "Fiscal Year"]])

df_plot12_polar = df_plot12_polar.groupby(
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

df_plot12_polar = df_plot12_polar.set_index('Fiscal Year')
chart_totaldebt_on_country = sns.lineplot(data=df_plot12_polar)
chart_totaldebt_on_country.set_xticklabels(chart_totaldebt_on_country.get_xticklabels(), rotation=45)
chart_totaldebt_on_country  # .set_ylim(0,400000000)
plt.pyplot.show()

## PLOT 13 DENMARK ###

df_plot13 = combined_dataset[
    ['Fiscal Year', 'Bonds and Notes', "Term Loans", "Revolving Credit", "Other Borrowings", "Capital Lease",
     "Commercial Paper", "Trust Preferred", "Country of Exchange"]]
df_plot13 = combined_dataset.loc[
    combined_dataset['Country of Exchange'] == 'Denmark', ['Fiscal Year', 'Bonds and Notes', "Term Loans",
                                                           "Revolving Credit", "Other Borrowings", "Capital Lease",
                                                           "Commercial Paper", "Trust Preferred"]]
df_plot13_polar = pl.from_pandas(
    df_plot13[
        ['Term Loans', 'Bonds and Notes', 'Revolving Credit', 'Other Borrowings', 'Capital Lease', 'Commercial Paper',
         'Trust Preferred', "Fiscal Year"]])

df_plot13_polar = df_plot13_polar.groupby(
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

df_plot13_polar = df_plot13_polar.set_index('Fiscal Year')
chart_totaldebt_on_country = sns.lineplot(data=df_plot13_polar)
chart_totaldebt_on_country.set_xticklabels(chart_totaldebt_on_country.get_xticklabels(), rotation=45)
chart_totaldebt_on_country  # .set_ylim(0,400000000)
plt.pyplot.show()

## PLOT 14 FINLAND ###

df_plot14 = combined_dataset[
    ['Fiscal Year', 'Bonds and Notes', "Term Loans", "Revolving Credit", "Other Borrowings", "Capital Lease",
     "Commercial Paper", "Trust Preferred", "Country of Exchange"]]
df_plot14 = combined_dataset.loc[
    combined_dataset['Country of Exchange'] == 'Finland', ['Fiscal Year', 'Bonds and Notes', "Term Loans",
                                                           "Revolving Credit", "Other Borrowings", "Capital Lease",
                                                           "Commercial Paper", "Trust Preferred"]]
df_plot14_polar = pl.from_pandas(
    df_plot14[
        ['Term Loans', 'Bonds and Notes', 'Revolving Credit', 'Other Borrowings', 'Capital Lease', 'Commercial Paper',
         'Trust Preferred', "Fiscal Year"]])

df_plot14_polar = df_plot14_polar.groupby(
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

df_plot14_polar = df_plot14_polar.set_index('Fiscal Year')
chart_totaldebt_on_country = sns.lineplot(data=df_plot14_polar)
chart_totaldebt_on_country.set_xticklabels(chart_totaldebt_on_country.get_xticklabels(), rotation=45)
chart_totaldebt_on_country  # .set_ylim(0,400000000)
plt.pyplot.show()

chart_debt_totalpct = df_plot2_debt_totalpct_polar.set_index('year').sort_index(ascending=True).plot(kind='bar',
                                                                                                     stacked=True)
chart_debt_totalpct.set_xticklabels(chart_debt_totalpct.get_xticklabels(), rotation='vertical')

### PLOT 15 ###

df_plot15 = combined_dataset[['year', 'Bonds and Notes', 'Country of Exchange']]
df_plot15_polar = pl.from_pandas(
    df_plot15[['year', 'Bonds and Notes', 'Country of Exchange']])

df_plot15 = df_plot15_polar.groupby(
    [
        "year", 'Country of Exchange'
    ]
).agg(
    [
        pl.sum("Bonds and Notes").alias("Bonds and Notes Annually")
    ]
).to_pandas()

df_plot15 = df_plot15.set_index('year')
df_plot15 = df_plot15.fillna(0)
chart15 = sns.lineplot(data=df_plot15, x=df_plot15.index, y='Bonds and Notes Annually', hue='Country of Exchange')
chart15.set_xticklabels(chart15.get_xticklabels(), rotation=45)
plt.pyplot.show()

### PLOT 16 ###

df_plot16 = combined_dataset[['year', "Term Loans", 'Country of Exchange']]
df_plot16_polar = pl.from_pandas(
    df_plot16[['year', "Term Loans", 'Country of Exchange']])

df_plot16 = df_plot16_polar.groupby(
    [
        "year", 'Country of Exchange'
    ]
).agg(
    [
        pl.sum("Term Loans").alias("Term Loans Annually")
    ]
).to_pandas()

df_plot16 = df_plot16.set_index('year')
df_plot16 = df_plot16.fillna(0)
chart16 = sns.lineplot(data=df_plot16, x=df_plot16.index, y='Term Loans Annually', hue='Country of Exchange')
chart16.set_xticklabels(chart16.get_xticklabels(), rotation=45)
plt.pyplot.show()

### PLOT 17 ###

df_plot17 = combined_dataset[['year', "Revolving Credit", 'Country of Exchange']]
df_plot17_polar = pl.from_pandas(
    df_plot17[['year', "Revolving Credit", 'Country of Exchange']])

df_plot17 = df_plot17_polar.groupby(
    [
        "year", 'Country of Exchange'
    ]
).agg(
    [
        pl.sum("Revolving Credit").alias("Revolving Credit Annually")
    ]
).to_pandas()

df_plot17 = df_plot17.set_index('year')
df_plot17 = df_plot17.fillna(0)
chart17 = sns.lineplot(data=df_plot17, x=df_plot17.index, y='Revolving Credit Annually', hue='Country of Exchange')
chart17.set_xticklabels(chart17.get_xticklabels(), rotation=45)
plt.pyplot.show()

### PLOT 18 ###

df_plot18 = combined_dataset[['year', "Other Borrowings", 'Country of Exchange']]
df_plot18_polar = pl.from_pandas(
    df_plot18[['year', "Other Borrowings", 'Country of Exchange']])

df_plot18 = df_plot18_polar.groupby(
    [
        "year", 'Country of Exchange'
    ]
).agg(
    [
        pl.sum("Other Borrowings").alias("Other Borrowings Annually")
    ]
).to_pandas()

df_plot18 = df_plot18.set_index('year')
df_plot18 = df_plot18.fillna(0)
chart18 = sns.lineplot(data=df_plot18, x=df_plot18.index, y='Other Borrowings Annually', hue='Country of Exchange')
chart18.set_xticklabels(chart18.get_xticklabels(), rotation=45)
plt.pyplot.show()

### PLOT 19 ###

df_plot19 = combined_dataset[['year', "Capital Lease", 'Country of Exchange']]
df_plot19_polar = pl.from_pandas(
    df_plot19[['year', "Capital Lease", 'Country of Exchange']])

df_plot19 = df_plot19_polar.groupby(
    [
        "year", 'Country of Exchange'
    ]
).agg(
    [
        pl.sum("Capital Lease").alias("Capital Lease Annually")
    ]
).to_pandas()

df_plot19 = df_plot19.set_index('year')
df_plot19 = df_plot19.fillna(0)
chart19 = sns.lineplot(data=df_plot19, x=df_plot19.index, y='Capital Lease Annually', hue='Country of Exchange')
chart19.set_xticklabels(chart19.get_xticklabels(), rotation=45)
plt.pyplot.show()

### PLOT 20 ###

df_plot20 = combined_dataset[['year', "Commercial Paper", 'Country of Exchange']]
df_plot20_polar = pl.from_pandas(
    df_plot20[['year', "Commercial Paper", 'Country of Exchange']])

df_plot20 = df_plot20_polar.groupby(
    [
        "year", 'Country of Exchange'
    ]
).agg(
    [
        pl.sum("Commercial Paper").alias("Commercial Paper Annually")
    ]
).to_pandas()

df_plot20 = df_plot20.set_index('year')
df_plot20 = df_plot20.fillna(0)
chart20 = sns.lineplot(data=df_plot20, x=df_plot20.index, y='Commercial Paper Annually', hue='Country of Exchange')
chart20.set_xticklabels(chart20.get_xticklabels(), rotation=45)
plt.pyplot.show()

### PLOT 21 ###

df_plot21 = combined_dataset[['year', "Trust Preferred", 'Country of Exchange']]
df_plot21_polar = pl.from_pandas(
    df_plot21[['year', "Trust Preferred", 'Country of Exchange']])

df_plot21 = df_plot21_polar.groupby(
    [
        "year", 'Country of Exchange'
    ]
).agg(
    [
        pl.sum("Trust Preferred").alias("Trust Preferred Annually")
    ]
).to_pandas()

df_plot21 = df_plot21.set_index('year')
df_plot21 = df_plot21.fillna(0)
chart21 = sns.lineplot(data=df_plot21, x=df_plot21.index, y='Trust Preferred Annually', hue='Country of Exchange')
chart21.set_xticklabels(chart21.get_xticklabels(), rotation=45)
plt.pyplot.show()
