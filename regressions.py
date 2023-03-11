import pandas as pd
import statsmodels.api as sm


### reg1 multivariate regression?####
Y = combined_dataset['Total Debt CapitalIQ'].fillna(0)
X = combined_dataset[[
    'Bonds and Notes',
    "Trust Preferred",
    'Commercial Paper',
    "Capital Lease",
    "Other Borrowings",
    "Revolving Credit",
    "Term Loans"
]].fillna(0)
X = sm.add_constant(X)

reg1 = sm.OLS(Y, X)
reg1_res = reg1.fit()
print(reg1_res.summary())
