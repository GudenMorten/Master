import eikon as ek
import pandas as pd

# Replace YOUR_APP_KEY with your Eikon API key
ek.set_app_key('022971ea1eb44687b0bcb92bad73c9b83ab08a9d')

nordic_data = pd.read_excel("total_ISIN.xlsx")
list1 = nordic_data["International Security Identification Number"].dropna().to_list()


df2, _ = ek.get_data(list1, fields=['TR.F.STDebtCurrPortOfLTDebt.date', 'TR.F.STDebtCurrPortOfLTDebt',
                                    'TR.F.DebtNonConvertLT', 'TR.F.CapLeaseObligLT',
                                    'TR.F.CurrPortOfLTDebtExclCapLease', 'TR.F.CapLeaseCurrPort', 'TR.NAICSSectorCode',
                                    'TR.NAICSSector', 'TR.NAICSSubsectorCode', 'TR.NAICSSubsector',
                                    'TR.NAICSNationalIndustryCode', 'TR.NAICSNationalIndustry',
                                    'TR.NAICSIndustryGroupCode', 'TR.NAICSIndustryGroup'
                                    ],
                     parameters={'SDate': '1990-01-01', 'EDate': '2023-02-15', "Frq": "FY"})

print(df2.count())