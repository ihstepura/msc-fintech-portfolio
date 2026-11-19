# Data sources

The raw yield files are not included in this repository. Some come from investing.com, whose terms don't allow redistribution, and all can be downloaded from the sources below.

Save the files in `notebooks/data/` with the file names shown, since `1_load_yields.ipynb` looks for them there.

| Country | Series | Source | File name(s) |
|---|---|---|---|
| UK | Nominal zero-coupon spot curve (2y, 5y, 10y, 30y) | [Bank of England yield curves](https://www.bankofengland.co.uk/statistics/yield-curves) | `UK_GLC_Nominal_daily_data_2016_to_2024.xlsx`, `UK_GLC_Nominal_daily_data_2025_to_present.xlsx` |
| US | Constant-maturity Treasury yields (DGS2, DGS5, DGS10, DGS30) | [FRED](https://fred.stlouisfed.org) | `US_DGS2.xlsx`, `US_DGS5.xlsx`, `US_DGS10.xlsx`, `US_DGS30.xlsx` |
| Germany | Current-bond yields, 10y and 30y | [Deutsche Bundesbank](https://www.bundesbank.de/en/statistics) | `10_year_german.csv`, `30_year_german.csv` |
| France | Constant-maturity yields | [Banque de France](https://webstat.banque-france.fr) | `all_france_bond_maturities.csv` |
| Italy | 10y and 30y benchmark yields | [investing.com](https://www.investing.com/rates-bonds/) | `Italy_10-Year_Bond_Yield_Historical_Data.csv`, `Italy_30-Year_Bond_Yield_Historical_Data.csv` |
| Japan | 10y and 30y benchmark yields | [investing.com](https://www.investing.com/rates-bonds/) | `Japan_10-Year_Bond_Yield_Historical_Data.csv`, `Japan_30-Year_Bond_Yield_Historical_Data.csv` |
| Canada | 10y and 30y benchmark yields | [investing.com](https://www.investing.com/rates-bonds/) | `Canada_10-Year_Bond_Yield_Historical_Data.csv`, `Canada_30-Year_Bond_Yield_Historical_Data.csv` |

**Formatting notes** (handled in `1_load_yields.ipynb`):
- The two Bundesbank files have swapped names, so the loader identifies each one by its internal title rather than its file name.
- The Banque de France file uses semicolons as separators and commas as decimal points.
- The Canada 10y file uses MM/DD/YYYY dates, but the 30y file uses DD/MM/YYYY.
