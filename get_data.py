import eikon as ek

# Set the AppKey for the Eikon Data API
ek.set_app_key('62297b98ae4947f2ac01401801b0c32a165a1053')

# Retrieve data from the Eikon database
data, _ = ek.get_data('AAPL.O', ['TR.Revenue',])

# Print the data
print(data)