import eikon as ek
import pandas as pd

# Replace YOUR_APP_KEY with your Eikon API key
ek.set_app_key('022971ea1eb44687b0bcb92bad73c9b83ab08a9d')

nordic_data = pd.read_excel("total_ISIN.xlsx")
list1 = nordic_data["International Security Identification Number"].dropna().to_list()

df4, _ = ek.get_data(list1, fields=['TR.CommonName', 'TR.F.STDebtCurrPortOfLTDebt.date', 'TR.F.STDebtCurrPortOfLTDebt',
                                    'TR.F.DebtNonConvertLT', 'TR.F.CapLeaseObligLT',
                                    'TR.F.CurrPortOfLTDebtExclCapLease', 'TR.F.CapLeaseCurrPort', 'TR.NAICSSectorCode',
                                    'TR.NAICSSector', 'TR.NAICSSubsectorCode', 'TR.NAICSSubsector',
                                    'TR.NAICSNationalIndustryCode', 'TR.NAICSNationalIndustry',
                                    'TR.NAICSIndustryGroupCode', 'TR.NAICSIndustryGroup', 'TR.ExchangeCountry',
                                    'TR.ExchangeName', 'TR.F.TotDebtPctofTotEq', 'TR.F.DebtLTTot', 'TR.F.DebtTot',
                                    'TR.WACCCostofDebt', 'TR.F.MktCap', 'TR.F.NetIncAfterTax', 'TR.F.TotShHoldEq',
                                    'TR.EPSMean', 'TR.F.ComShrOutsTotDR', 'TR.PriceClose', 'TR.F.TotAssets',
                                    'TR.F.TotCurrAssets', 'TR.F.TotCurrLiab'
                                    ],
                     parameters={'SDate': '2001-01-01', 'EDate': '2023-12-31', "Frq": "FY"})

print(df4.count())


### Store data to csv file
df4.to_csv(r"C:\\Users\\morte\\PycharmProjects\\Master\\refinitivdata3.csv", index=False)

