import pandas as pd
import polars as pl

capitaliqquarterly = pd.read_parquet('capitaliq.parquet')
gvisin = pd.read_excel('gv-isin.xlsx')

# sort the data, getting only the values that are the most recent, not all duplicates as well
capitaliqquarterly = capitaliqquarterly[capitaliqquarterly["latestfilingforinstanceflag"].isin([1])]
capitaliqquarterly = capitaliqquarterly[capitaliqquarterly["latestforfinancialperiodflag"].isin([1])]

# make the "periodenddate" and 'filingdate' into date format
capitaliqquarterly['periodenddate'] = capitaliqquarterly['periodenddate'].apply(lambda x: pd.to_datetime(str(int(x))))
capitaliqquarterly['filingdate'] = capitaliqquarterly['filingdate'].apply(lambda x: pd.to_datetime(str(int(x))))

# creating and fixing certain columns
capitaliqquarterly['quarter'] = capitaliqquarterly['periodenddate'].dt.to_period('Q')
capitaliqquarterly['quarternumb'] = capitaliqquarterly['periodenddate'].dt.to_period('Q').dt.quarter
capitaliqquarterly['year'] = capitaliqquarterly['periodenddate'].dt.year
capitaliqquarterly['gvkey'] = capitaliqquarterly['gvkey'].apply(lambda x: (int(x)))

capitaliqquarterly.info()
# define a function to adjust the dataitemvalue based on unittypeid

def adjust_value(dataitemvalue, unittypeid):
    if unittypeid == 1:
        return dataitemvalue * 1000
    elif unittypeid == 2:
        return dataitemvalue * 1000000
    else:
        return dataitemvalue


# apply the function to create a new column with adjusted values
capitaliqquarterly['dataitemvalue'] = capitaliqquarterly.apply(lambda row: adjust_value(row['dataitemvalue'], row['unittypeid']), axis=1)



capitaliqannual = capitaliqquarterly[capitaliqquarterly['quarternumb'].isin([4])]

capitaliqannual_summary = pl.from_pandas(
    capitaliqannual[["year", "gvkey", "capitalstructuresubtypeid", "dataitemvalue"]])

capitaliqannual_summary = capitaliqannual_summary.groupby(
    [
        "year",
        "gvkey",
        "capitalstructuresubtypeid"
    ]
).agg(
    [
        pl.sum("dataitemvalue").alias("Value")
    ]
)

capitalstructure_sorted = capitaliqannual_summary.pivot(
    values="Value",
    index=["year", "gvkey"],
    columns="capitalstructuresubtypeid"
).to_pandas()

merged_data = pd.merge(capitaliqannual, gvisin, on='gvkey')

