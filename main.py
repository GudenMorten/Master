import pandas as pd

dataf = pd.read_csv('gathered_data.csv')

# fixing the dataframe
dataf[['Company Common Name', 'NAICS Sector Code', 'NAICS Sector Name',
    'NAICS Subsector Code', 'NAICS Subsector Name', 'NAICS National Industry Code',
    'NAICS National Industry Name', 'NAICS Industry Group Code', 'NAICS Industry Group Name',
    'Country of Exchange', 'Exchange Name']] = dataf.groupby('Instrument')[['Company Common Name', 'NAICS Sector Code',
    'NAICS Sector Name', 'NAICS Subsector Code', 'NAICS Subsector Name', 'NAICS National Industry Code',
    'NAICS National Industry Name', 'NAICS Industry Group Code', 'NAICS Industry Group Name',
    'Country of Exchange', 'Exchange Name']].fillna(method='ffill')

