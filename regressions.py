import pandas as pd
import statsmodels.api as sm


### reg1 multivariate regression?####
reg1 = combined_dataset[['Bonds and Notes', "NAICS Sector Code", 'Total Debt CapitalIQ', 'ROE', 'year', 'Market Capitalization']]
dummies_NAICS = pd.get_dummies(reg1['NAICS Sector Code'])
dummies_year = pd.get_dummies(reg1['year'])
reg1_merge = pd.concat([reg1, dummies_NAICS, dummies_year], axis=1)
reg1_merge.drop(['NAICS Sector Code', 'year'], axis=1, inplace=True)
Y = reg1_merge['Bonds and Notes'].fillna(0)
X = reg1_merge.drop('Bonds and Notes', axis=1).fillna(0)
X.columns = ['Total Debt CapitalIQ', 'ROE', 'Market Capitalization', '11', '21', '23', '31-33', '42', '44-45', '48-49', '51', '54', '55', '56', '61',    '62',    '71',    '72',    '81', 'year_2000',   'year_2001',    'year_2002',    'year_2003',    'year_2004',    'year_2005',    'year_2006',    'year_2007',    'year_2008',    'year_2009',    'year_2010',    'year_2011',    'year_2012',    'year_2013',    'year_2014',    'year_2015',    'year_2016',    'year_2017',    'year_2018',    'year_2019',    'year_2020',    'year_2021']
X = X[[
    'Market Capitalization',
    '21',
    '23',
    '31-33',
    '42',
    '44-45',
    '48-49',
    '51',
    '54',
    '55',
    '56',
    '61',
    '62',
    '71',
    '72',
    '81',
    'year_2000',
    'year_2001',
    'year_2002',
    'year_2003',
    'year_2004',
    'year_2005',
    'year_2006',
    'year_2007',
    'year_2008',
    'year_2009',
    'year_2010',
    'year_2011',
    'year_2012',
    'year_2013',
    'year_2014',
    'year_2015',
    'year_2016',
    'year_2017',
    'year_2018',
    'year_2019',
    'year_2020',
    'year_2021'
]].fillna(0)
X = sm.add_constant(X)

reg1 = sm.OLS(Y, X)
reg1_res = reg1.fit()
print(reg1_res.summary())