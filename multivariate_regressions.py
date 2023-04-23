import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from censreg import tobit

### Multivariate regression
multivariate_reg = spec_data_needed.copy()
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy'], axis=1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date', 'NAICS Sector Code']],
                            how='left', on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
## adding year and industry fixed effects
dummies_NAICS = pd.get_dummies(multivariate_merge['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge['Date'])
multivariate_merge = pd.concat([multivariate_merge, dummies_NAICS, dummies_year], axis=1)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date', 'NAICS Sector Code', '31-33'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)

Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y', 2001, 2021, '11'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg = sm.OLS(Y, X)
multivariate_reg_res = multivariate_reg.fit(cov_type="HC0")
print(multivariate_reg_res.summary())

new_cols = {}
for col in current_cols:
    if isinstance(col, str):
        new_cols[col] = col.replace('_', ' ')
    else:
        new_cols[col] = str(col)

multivariate_merge = multivariate_merge.rename(columns=new_cols)

# check VIF
X = multivariate_merge.drop(['HHI y', '2001', '2021', '11'], axis=1).fillna(0)
y = multivariate_merge['HHI y'].fillna(0)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_std, i) for i in range(X_std.shape[1])]
vif["features"] = X.columns
print(vif)

#########################

### Multivariate regression base 5 variables
multivariate_reg = spec_data_needed.copy()
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy', 'Advertising', 'CAPEX', 'Book Leverage'], axis=1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date', 'NAICS Sector Code']],
                            how='left', on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
## adding year and industry fixed effects
dummies_NAICS = pd.get_dummies(multivariate_merge['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge['Date'])
multivariate_merge = pd.concat([multivariate_merge, dummies_NAICS, dummies_year], axis=1)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date', 'NAICS Sector Code', '31-33'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)

Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y', 2001, 2021, '11'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg = sm.OLS(Y, X)
multivariate_reg_res = multivariate_reg.fit(cov_type="HC0")
print(multivariate_reg_res.summary())

########################### Adding book leverage

multivariate_reg = spec_data_needed.copy()
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy', 'Advertising', 'CAPEX'], axis=1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date', 'NAICS Sector Code']],
                            how='left', on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
## adding year and industry fixed effects
dummies_NAICS = pd.get_dummies(multivariate_merge['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge['Date'])
multivariate_merge = pd.concat([multivariate_merge, dummies_NAICS, dummies_year], axis=1)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date', 'NAICS Sector Code', '31-33'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)

Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y', 2001, 2021, '11'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg = sm.OLS(Y, X)
multivariate_reg_res = multivariate_reg.fit(cov_type="HC0")
print(multivariate_reg_res.summary())

####################### Adding CAPEX

multivariate_reg = spec_data_needed.copy()
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy', 'Advertising'], axis=1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date', 'NAICS Sector Code']],
                            how='left', on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
## adding year and industry fixed effects
dummies_NAICS = pd.get_dummies(multivariate_merge['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge['Date'])
multivariate_merge = pd.concat([multivariate_merge, dummies_NAICS, dummies_year], axis=1)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date', 'NAICS Sector Code', '31-33'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)

Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y', 2001, 2021, '11'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg = sm.OLS(Y, X)
multivariate_reg_res = multivariate_reg.fit(cov_type="HC0")
print(multivariate_reg_res.summary())

################ Adding advertising
multivariate_reg = spec_data_needed.copy()
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy'], axis=1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date', 'NAICS Sector Code']],
                            how='left', on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
## adding year and industry fixed effects
dummies_NAICS = pd.get_dummies(multivariate_merge['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge['Date'])
multivariate_merge = pd.concat([multivariate_merge, dummies_NAICS, dummies_year], axis=1)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date', 'NAICS Sector Code', '31-33'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)

Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y', 2001, 2021, '11'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg = sm.OLS(Y, X)
multivariate_reg_res = multivariate_reg.fit(cov_type="HC0")
print(multivariate_reg_res.summary())

######################## (5)-(8) som er inkludert country of exchange FE


#################### (5)
multivariate_reg = spec_data_needed.copy()
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy', 'Advertising', 'CAPEX', 'Book Leverage'], axis=1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date', 'NAICS Sector Code', 'Country of Exchange']],
                            how='left', on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
## adding year and industry fixed effects
dummies_NAICS = pd.get_dummies(multivariate_merge['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge['Date'])
dummies_country = pd.get_dummies(multivariate_merge['Country of Exchange'])
multivariate_merge = pd.concat([multivariate_merge, dummies_NAICS, dummies_year, dummies_country], axis=1)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date', 'NAICS Sector Code', 'Country of Exchange', '31-33'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)

Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y', 2001, 2021, '11', 'Denmark'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg = sm.OLS(Y, X)
multivariate_reg_res = multivariate_reg.fit(cov_type="HC0")
print(multivariate_reg_res.summary())


############################# (6)
multivariate_reg = spec_data_needed.copy()
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy', 'Advertising', 'CAPEX'], axis=1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date', 'NAICS Sector Code', 'Country of Exchange']],
                            how='left', on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
## adding year and industry fixed effects
dummies_NAICS = pd.get_dummies(multivariate_merge['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge['Date'])
dummies_country = pd.get_dummies(multivariate_merge['Country of Exchange'])
multivariate_merge = pd.concat([multivariate_merge, dummies_NAICS, dummies_year, dummies_country], axis=1)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date', 'NAICS Sector Code', 'Country of Exchange', '31-33'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)

Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y', 2001, 2021, '11', 'Denmark'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg = sm.OLS(Y, X)
multivariate_reg_res = multivariate_reg.fit(cov_type="HC0")
print(multivariate_reg_res.summary())

############################# (7)
multivariate_reg = spec_data_needed.copy()
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy', 'Advertising'], axis=1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date', 'NAICS Sector Code', 'Country of Exchange']],
                            how='left', on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
## adding year and industry fixed effects
dummies_NAICS = pd.get_dummies(multivariate_merge['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge['Date'])
dummies_country = pd.get_dummies(multivariate_merge['Country of Exchange'])
multivariate_merge = pd.concat([multivariate_merge, dummies_NAICS, dummies_year, dummies_country], axis=1)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date', 'NAICS Sector Code', 'Country of Exchange', '31-33'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)

Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y', 2001, 2021, '11', 'Denmark'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg = sm.OLS(Y, X)
multivariate_reg_res = multivariate_reg.fit(cov_type="HC0")
print(multivariate_reg_res.summary())

############################# (8)
multivariate_reg = spec_data_needed.copy()
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy'], axis=1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date', 'NAICS Sector Code', 'Country of Exchange']],
                            how='left', on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
## adding year and industry fixed effects
dummies_NAICS = pd.get_dummies(multivariate_merge['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge['Date'])
dummies_country = pd.get_dummies(multivariate_merge['Country of Exchange'])
multivariate_merge = pd.concat([multivariate_merge, dummies_NAICS, dummies_year, dummies_country], axis=1)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date', 'NAICS Sector Code', 'Country of Exchange', '31-33'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)

Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y', 2001, 2021, '11', 'Denmark'], axis=1).fillna(0)

X = sm.add_constant(X)
multivariate_reg = sm.OLS(Y, X)
multivariate_reg_res = multivariate_reg.fit(cov_type="HC0")
print(multivariate_reg_res.summary())

new_cols = {}
for col in current_cols:
    if isinstance(col, str):
        new_cols[col] = col.replace('_', ' ')
    else:
        new_cols[col] = str(col)

multivariate_merge = multivariate_merge.rename(columns=new_cols)

# check VIF
X = multivariate_merge.drop(['HHI y', '2001', '2021', '11', 'Denmark'], axis=1).fillna(0)
y = multivariate_merge['HHI y'].fillna(0)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_std, i) for i in range(X_std.shape[1])]
vif["features"] = X.columns
print(vif)


import tobit_reg as tob
multivariate_reg = spec_data_needed.copy()
multivariate_reg = multivariate_reg.drop(['1st T', '3rd T', 'Dividend Payer', 'ln Sales', 'DS90 dummy'], axis=1)
multivariate_reg = pd.merge(multivariate_reg, which_firms_specialize[['Instrument', 'Date', 'NAICS Sector Code', 'Country of Exchange']],
                            how='left', on=['Instrument', 'Date'])
multivariate_reg['Date'] = multivariate_reg['Date'].dt.year
## lagging HHI by -1
multivariate_reg.set_index(['Date', 'Instrument'], inplace=True)
shifted = multivariate_reg.groupby(level='Instrument').shift(-1)
multivariate_reg.join(shifted.rename(columns=lambda x: x + '_lag'))
multivariate_merge = pd.merge(multivariate_reg, shifted['HHI'], left_index=True, right_index=True, how='left')
multivariate_merge.reset_index(inplace=True)
## adding year and industry fixed effects
dummies_NAICS = pd.get_dummies(multivariate_merge['NAICS Sector Code'])
dummies_year = pd.get_dummies(multivariate_merge['Date'])
dummies_country = pd.get_dummies(multivariate_merge['Country of Exchange'])
multivariate_merge = pd.concat([multivariate_merge, dummies_NAICS, dummies_year, dummies_country], axis=1)
multivariate_merge.drop(['HHI_x', 'Instrument', 'Date', 'NAICS Sector Code', 'Country of Exchange', '31-33'], axis=1, inplace=True)
multivariate_merge.dropna(inplace=True)

Y = multivariate_merge['HHI_y'].fillna(0)
X = multivariate_merge.drop(['HHI_y', 2001, 2021, '11', 'Denmark'], axis=1).fillna(0)

tob.Tobit(Y, X)