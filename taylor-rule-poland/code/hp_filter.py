"""
Hodrick-Prescott filter for Polish real GDP.

Reads quarterly real GDP from real_gdp.xlsx, applies the HP filter
(lambda = 1600) to log GDP, and writes the HP trend and the output gap,
ygap = 100 * (ln GDP - HP trend), to real_gdp_with_hp.xlsx.

The output gap feeds column E of the Poland_Data sheet in
model/poland_taylor_rule_regression.xlsx.
"""

from pathlib import Path

import numpy as np
import openpyxl
from statsmodels.tsa.filters.hp_filter import hpfilter


HERE        = Path(__file__).parent
INFILE      = HERE / "real_gdp.xlsx"
SHEET       = "Sheet1"
GDP_COL     = 1
HEADER_ROWS = 1
LAMBDA      = 1600
OUTFILE     = HERE / "real_gdp_with_hp.xlsx"

wb = openpyxl.load_workbook(INFILE, data_only=True)
ws = wb[SHEET]

rows = []
gdp  = []
for r in range(HEADER_ROWS + 1, ws.max_row + 1):
    v = ws.cell(r, GDP_COL).value
    if v is None:
        continue
    rows.append(r)
    gdp.append(float(v))

gdp = np.asarray(gdp, dtype=float)


log_gdp = np.log(gdp)
cycle, trend = hpfilter(log_gdp, lamb=LAMBDA)
gap = 100.0 * (log_gdp - trend)


tcol = GDP_COL + 1
gcol = GDP_COL + 2
ws.cell(HEADER_ROWS, tcol, "HP trend")
ws.cell(HEADER_ROWS, gcol, "ygap (%)")
for r, tr, gp in zip(rows, trend, gap):
    ws.cell(r, tcol, round(float(tr), 6))
    ws.cell(r, gcol, round(float(gp), 2))

wb.save(OUTFILE)


print(f"Read {len(gdp)} GDP values from {INFILE}. lambda = {LAMBDA}")
print(f"{'row':>4} {'GDP':>12} {'HP trend':>10} {'gap %':>7}")
for r, g, tr, gp in zip(rows, gdp, trend, gap):
    print(f"{r:>4} {g:>12.2f} {tr:>10.6f} {gp:>7.2f}")
print(f"\nSaved -> {OUTFILE}")
