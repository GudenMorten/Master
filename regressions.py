import pandas as pd
import statsmodels.api as sm


### reg1 multivariate regression?####
reg1 = combined_dataset[['Bonds and Notes',
    "Trust Preferred",
    'Commercial Paper',
    "Capital Lease",
    "Other Borrowings",
    "Revolving Credit",
    "Term Loans",
    "NAICS Sector Code",
    'Total Debt CapitalIQ']]
dummies = pd.get_dummies(reg1['NAICS Sector Code'])
reg1_merge = pd.merge(reg1, dummies, left_index=True, right_index=True)
Y = reg1_merge['Total Debt CapitalIQ'].fillna(0)
X = reg1_merge[[
    #'Bonds and Notes',
    #"Trust Preferred",
    #'Commercial Paper',
    #"Capital Lease",
    #"Other Borrowings",
    #"Revolving Credit",
    #"Term Loans",
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
    '81'
]].fillna(0)

X = sm.add_constant(X)

reg1 = sm.OLS(Y, X)
reg1_res = reg1.fit()
print(reg1_res.summary())
