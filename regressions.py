import pandas as pd
import statsmodels.api as sm



### reg1 multivariate regression?####
reg1 = combined_dataset[
    ['Bonds and Notes/Total Debt', "NAICS Sector Code", 'ROE', 'year']]
dummies_NAICS = pd.get_dummies(reg1['NAICS Sector Code'])
dummies_year = pd.get_dummies(reg1['year'])
reg1_merge = pd.concat([reg1, dummies_NAICS, dummies_year], axis=1)
reg1_merge.drop(['NAICS Sector Code', 'year'], axis=1, inplace=True)
Y = reg1_merge['Bonds and Notes/Total Debt'].fillna(0)
X = reg1_merge.drop('Bonds and Notes/Total Debt', axis=1).fillna(0)
X.columns = ['ROE', '11', '21', '23', '31-33', '42', '44-45', '48-49',
             '51', '54', '55', '56', '61', '62', '71', '72', '81', 'year_2000', 'year_2001', 'year_2002', 'year_2003',
             'year_2004', 'year_2005', 'year_2006', 'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011',
             'year_2012', 'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018', 'year_2019',
             'year_2020', 'year_2021']
X = X[['ROE', '21', '23', '31-33', '42', '44-45', '48-49',
       '51', '54', '55', '56', '61', '62', '71', '72', '81', 'year_2000', 'year_2001', 'year_2002', 'year_2003',
       'year_2004', 'year_2005', 'year_2006', 'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011',
       'year_2012', 'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018', 'year_2019',
       'year_2020', 'year_2021']].fillna(0)
X = sm.add_constant(X)

reg1 = sm.OLS(Y, X)
reg1_res = reg1.fit()
print(reg1_res.summary())

## check for multicollinearity with chi-square test ##
from statsmodels.stats.outliers_influence import variance_inflation_factor
df = reg1_merge.fillna(0)
# Select the columns you want to check for multicollinearity
X = reg1_merge.drop('Bonds and Notes', axis=1).fillna(0)
X.columns = ['Total Debt CapitalIQ', 'ROE', 'Market Capitalization', '11', '21', '23', '31-33', '42', '44-45', '48-49',
             '51', '54', '55', '56', '61', '62', '71', '72', '81', 'year_2000', 'year_2001', 'year_2002', 'year_2003',
             'year_2004', 'year_2005', 'year_2006', 'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011',
             'year_2012', 'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018', 'year_2019',
             'year_2020', 'year_2021']
X = X[['ROE', 'Market Capitalization', '21', '23', '31-33', '42', '44-45', '48-49',
       '51', '54', '55', '56', '61', '62', '71', '72', '81', 'year_2000', 'year_2001', 'year_2002', 'year_2003',
       'year_2004', 'year_2005', 'year_2006', 'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011',
       'year_2012', 'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018', 'year_2019',
       'year_2020', 'year_2021']].fillna(0)
#cols_to_check = X
#
## Calculate the VIF for each column
#vif = pd.DataFrame()
#vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(len(cols_to_check))]
#vif["features"] = cols_to_check
#
## Print the results
#print(vif)

############ REG 2 ###############
reg2 = combined_dataset[['Bonds and Notes', "NAICS Sector Code", 'ROE', 'year',
                         'Market Capitalization', 'Country of Exchange']]
dummies_NAICS2 = pd.get_dummies(reg2['NAICS Sector Code'])
dummies_year2 = pd.get_dummies(reg2['year'])
dummies_CoE = pd.get_dummies(reg2['Country of Exchange'])
reg2_merge = pd.concat([reg2, dummies_NAICS2, dummies_year2, dummies_CoE], axis=1)
reg2_merge.drop(['NAICS Sector Code', 'year', 'Country of Exchange'], axis=1, inplace=True)
Y = reg2_merge['Bonds and Notes'].fillna(0)
Y.columns = ['Bonds and Notes']
X = reg2_merge.drop('Bonds and Notes', axis=1)
X.columns = ['ROE', 'Market Capitalization', '11', '21', '23',
             '31-33', '42', '44-45', '48-49', '51', '54', '55', '56', '61', '62', '71', '72', '81', 'year_2000',
             'year_2001', 'year_2002', 'year_2003', 'year_2004', 'year_2005', 'year_2006',
             'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011', 'year_2012',
             'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018',
             'year_2019', 'year_2020', 'year_2021', 'Denmark', 'Finland', 'Norway', 'Sweden']
X = X[['ROE', 'Market Capitalization', '11', '21', '23', '31-33',
       '42', '44-45', '48-49', '51', '54', '55', '56', '61', '62', '71', '72', '81',
       'year_2001', 'year_2002', 'year_2003', 'year_2004', 'year_2005', 'year_2006',
       'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011', 'year_2012',
       'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018',
       'year_2019', 'year_2020', 'year_2021', 'Denmark', 'Finland', 'Norway', 'Sweden']].fillna(0)
X = sm.add_constant(X)

reg2 = sm.OLS(Y, X)
reg2_res = reg2.fit()
print(reg2_res.summary())

############ REG 3 ###############
reg3 = combined_dataset[['Bonds and Notes', 'Total Debt CapitalIQ', 'ROE', 'year',
                         'Market Capitalization', 'Country of Exchange']]
dummies_year = pd.get_dummies(reg3['year'])
dummies_CoE = pd.get_dummies(reg3['Country of Exchange'])
reg3_merge = pd.concat([reg3, dummies_year, dummies_CoE], axis=1)
reg3_merge.drop(['year', 'Country of Exchange'], axis=1, inplace=True)
Y = reg3_merge['Bonds and Notes'].fillna(0)
Y.columns = ['Bonds and Notes']
X = reg3_merge.drop('Bonds and Notes', axis=1)
X.columns = ['Total Debt CapitalIQ', 'ROE', 'Market Capitalization', 'year_2000',
             'year_2001', 'year_2002', 'year_2003', 'year_2004', 'year_2005', 'year_2006',
             'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011', 'year_2012',
             'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018',
             'year_2019', 'year_2020', 'year_2021', 'Denmark', 'Finland', 'Norway', 'Sweden']
X = X[['Total Debt CapitalIQ', 'ROE', 'Market Capitalization',
       'year_2001', 'year_2002', 'year_2003', 'year_2004', 'year_2005', 'year_2006',
       'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011', 'year_2012',
       'year_2013', 'year_2014', 'year_2015', 'year_2016', 'year_2017', 'year_2018',
       'year_2019', 'year_2020', 'year_2021', 'Denmark', 'Finland', 'Norway', 'Sweden']].fillna(0)
X = sm.add_constant(X)

reg3 = sm.OLS(Y, X)
reg3_res = reg3.fit()
print(reg3_res.summary())

############# reg 4 ##############
reg4 = combined_dataset[['Bonds and Notes', 'ROE', 'year', 'Market Capitalization', 'Country of Exchange']]
reg4_norway = reg4.loc[reg4['Country of Exchange'] == 'Norway'].loc[:,
              ['Bonds and Notes', 'ROE', 'year', 'Market Capitalization']].fillna(0)
reg4_denmark = reg4.loc[reg4['Country of Exchange'] == 'Denmark'].loc[:,
               ['Bonds and Notes', 'ROE', 'year', 'Market Capitalization']].fillna(0)
reg4_sweden = reg4.loc[reg4['Country of Exchange'] == 'Sweden'].loc[:,
              ['Bonds and Notes', 'ROE', 'year', 'Market Capitalization']].fillna(0)
reg4_finland = reg4.loc[reg4['Country of Exchange'] == 'Finland'].loc[:,
               ['Bonds and Notes', 'ROE', 'year', 'Market Capitalization']].fillna(0)

## dummies ##
dummies_year = pd.get_dummies(reg4['year'])

## merging dataframes ##
reg4_norway = pd.concat([reg4_norway, dummies_year], axis=1)
reg4_denmark = pd.concat([reg4_denmark, dummies_year], axis=1)
reg4_sweden = pd.concat([reg4_sweden, dummies_year], axis=1)
reg4_finland = pd.concat([reg4_finland, dummies_year], axis=1)

## Y ##
Y1 = reg4_norway['Bonds and Notes'].fillna(0)
Y2 = reg4_denmark['Bonds and Notes'].fillna(0)
Y3 = reg4_sweden['Bonds and Notes'].fillna(0)
Y4 = reg4_finland['Bonds and Notes'].fillna(0)

## removing dupes ##
reg4_norway.drop(['year', 'Bonds and Notes'], inplace=True, axis=1)
reg4_denmark.drop(['year', 'Bonds and Notes'], inplace=True, axis=1)
reg4_sweden.drop(['year', 'Bonds and Notes'], inplace=True, axis=1)
reg4_finland.drop(['year', 'Bonds and Notes'], inplace=True, axis=1)

## X ##
X1 = reg4_norway.fillna(0)
X2 = reg4_denmark.fillna(0)
X3 = reg4_sweden.fillna(0)
X4 = reg4_finland.fillna(0)

reg4_norway = sm.OLS(Y1, X1)
reg4_denmark = sm.OLS(Y2, X2)
reg4_sweden = sm.OLS(Y3, X3)
reg4_finland = sm.OLS(Y4, X4)

reg4_norway_res = reg4_norway.fit()
reg4_denmark_res = reg4_denmark.fit()
reg4_sweden_res = reg4_sweden.fit()
reg4_finland_res = reg4_finland.fit()

print(reg4_norway_res.summary())
print(reg4_denmark_res.summary())
print(reg4_sweden_res.summary())
print(reg4_finland_res.summary())
