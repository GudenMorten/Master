import eikon as ek
import pandas as pd

# Replace YOUR_APP_KEY with your Eikon API key
ek.set_app_key('022971ea1eb44687b0bcb92bad73c9b83ab08a9d')

nordic_data = pd.read_excel("total_ISIN.xlsx")
nordic_data2 = pd.read_excel("ISIN2.xlsx")
list1 = nordic_data["International Security Identification Number"].dropna().to_list()
list2 = nordic_data2["International Security Identification Number"].dropna().to_list()

df5, _ = ek.get_data(list2, fields=['TR.CommonName', 'TR.F.STDebtCurrPortOfLTDebt.date', 'TR.F.STDebtCurrPortOfLTDebt',
                                    'TR.F.DebtNonConvertLT', 'TR.F.CapLeaseObligLT',
                                    'TR.F.CurrPortOfLTDebtExclCapLease', 'TR.F.CapLeaseCurrPort', 'TR.NAICSSectorCode',
                                    'TR.NAICSSubsectorCode',
                                    'TR.NAICSNationalIndustryCode',
                                    'TR.NAICSIndustryGroupCode', 'TR.ExchangeCountry',
                                    'TR.F.TotDebtPctofTotEq', 'TR.F.DebtLTTot', 'TR.F.DebtTot',
                                    'TR.F.MktCap', 'TR.F.NetIncAfterTax', 'TR.F.TotShHoldEq',
                                    'TR.EPSMean', 'TR.F.ComShrOutsTotDR', 'TR.PriceClose', 'TR.F.TotAssets',
                                    'TR.F.TotCurrAssets', 'TR.F.TotCurrLiab'
                                    ],
                     parameters={'SDate': '2001-01-01', 'EDate': '2023-12-31', "Frq": "FY"})

print(df5.count())

datatest, _ = ek.get_data(list1, fields=['TR.IssuerRating', 'TR.FiFitchsRating', 'TR.FiMoodysRating', 'TR.FiSPRating'],
                     parameters={'SDate': '2001-01-01', 'EDate': '2023-12-31', "Frq": "FY",})

datatest2, _ = ek.get_data(list2, fields=['TR.IssuerRating', 'TR.FiFitchsRating', 'TR.FiMoodysRating', 'TR.FiSPRating'],
                     parameters={'SDate': '2001-01-01', 'EDate': '2023-12-31', "Frq": "FY",})


### Store data to csv file
df5.to_csv(r"C:\\Users\\morte\\PycharmProjects\\Master\\refinitivdata5.csv", index=False)
assetsusd.to_csv(r"C:\\Users\\morte\\PycharmProjects\\Master\\assetsusd.csv", index=False)
