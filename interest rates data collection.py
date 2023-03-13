import requests
import pandas as pd

############# FINLAND ###############
# Set the API endpoint and parameters
url = 'https://www.suomenpankki.fi/wp-content/plugins/shortcode-tables/api/data.php'
params = {
    'type': 'px-web-table',
    'queryid': 'BOF_3_3_EN',
    'language': 'en',
    'vs': '201',
    'yp': '20',
    'cube': 'BOF_3_3_EN',
    'dims': 'COUNTRY-8.FREQ-5.PUBIND-2.RATE-5.SECURITY-2',
    'showcell': 'code',
    'content': 'csv'
}

# Send a GET request to the API endpoint with the parameters
response = requests.get(url, params=params)

# Parse the response as a Pandas DataFrame
df = pd.read_csv(response.content.decode('utf-8'))

# Filter the data to only include annual rates for Finland
fi_data = df.loc[(df['COUNTRY'] == 'FI') & (df['FREQ'] == 'A')]

# Rename the columns to more readable names
fi_data = fi_data.rename(columns={
    'TIME_PERIOD': 'Year',
    'RATE': 'Interest Rate'
})

# Convert the Year column to a datetime format
fi_data['Year'] = pd.to_datetime(fi_data['Year'], format='%YM%m')

# Set the Year column as the index
fi_data = fi_data.set_index('Year')

# Print the resulting DataFrame
print(fi_data)

################### NORWAY ################
ir_norway = pd.read_csv('IR_norway.csv')