"""Pipeline v2 - step 3: observed interannual variability vs ENSO (ONI v6, NOAA CPC, fetched 2026-08-22).

Outputs: out/oni.csv, out/enso_corr.csv, out/annual_anomalies.csv
"""
import pickle
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT = 'pipe2/out'
ONI_TXT = """2010 1.5 1.2 0.9 0.4 -0.2 -0.7 -1.0 -1.2 -1.4 -1.5 -1.6 -1.5
2011 -1.3 -1.1 -0.8 -0.7 -0.5 -0.3 -0.4 -0.5 -0.8 -0.9 -1.0 -0.9
2012 -0.7 -0.6 -0.4 -0.3 -0.2 0.0 0.3 0.4 0.4 0.3 0.1 0.0
2013 -0.2 -0.3 -0.2 -0.2 -0.3 -0.4 -0.4 -0.3 -0.3 -0.1 -0.1 -0.1
2014 -0.2 -0.2 0.0 0.3 0.4 0.2 0.1 0.1 0.2 0.5 0.7 0.8
2015 0.7 0.7 0.7 0.9 1.0 1.2 1.4 1.7 2.0 2.3 2.5 2.6
2016 2.5 2.2 1.7 1.1 0.6 0.1 -0.2 -0.3 -0.4 -0.5 -0.5 -0.4
2017 -0.1 0.1 0.3 0.3 0.4 0.3 0.1 -0.1 -0.2 -0.4 -0.6 -0.8
2018 -0.7 -0.7 -0.5 -0.3 -0.1 0.1 0.2 0.3 0.5 0.8 1.0 1.1
2019 1.0 0.9 0.9 0.8 0.7 0.6 0.4 0.2 0.3 0.5 0.7 0.8
2020 0.7 0.7 0.6 0.3 0.0 -0.2 -0.3 -0.4 -0.8 -1.0 -1.1 -1.1
2021 -1.0 -0.9 -0.7 -0.6 -0.4 -0.3 -0.3 -0.5 -0.6 -0.8 -0.9 -0.8
2022 -0.8 -0.7 -0.8 -0.9 -0.8 -0.7 -0.7 -0.8 -0.9 -0.9 -0.8 -0.7
2023 -0.5 -0.3 -0.1 0.2 0.5 0.7 1.0 1.3 1.5 1.7 1.9 2.0
2024 1.8 1.5 1.2 0.8 0.4 0.2 0.1 0.0 -0.1 -0.2 -0.3 -0.4
2025 -0.5 -0.2 -0.1 0.0 0.0 0.0 -0.1 -0.3 -0.4 -0.6 -0.6 -0.6"""
rows = []
for line in ONI_TXT.splitlines():
    parts = line.split()
    y = int(parts[0])
    for m, v in enumerate(parts[1:], start=1):
        rows.append(dict(date=pd.Timestamp(y, m, 1), oni=float(v)))   # season centred on month m (DJF -> Jan, ...)
oni = pd.DataFrame(rows).set_index('date')['oni']
oni.to_csv(f'{OUT}/oni.csv')

P = pickle.load(open(f'{OUT}/prep.pkl', 'rb'))
obs = P['obs']
ORDER = ['SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'LN1', 'LN2', 'LN3', 'LN4', 'LN5', 'JC1', 'JC2']
anoms = {}
for st in ORDER:
    m = obs[st]['monthly']
    clim = obs[st]['clim']
    anoms[obs[st]['paper']] = 100 * (m - clim.reindex(m.index.month).values) / clim.reindex(m.index.month).values
A = pd.DataFrame(anoms)
A['CE'] = A[['SR1', 'SR2', 'SR3', 'SR4', 'SR5']].mean(axis=1)
A['RN'] = A[['LN1', 'LN2', 'LN3', 'LN4', 'LN5', 'JC1', 'JC2']].mean(axis=1)
A['ALL'] = A[[c for c in A.columns if c not in ('CE', 'RN')]].mean(axis=1)
A.to_csv(f'{OUT}/monthly_anomalies_pct.csv')

res = []
for col in A.columns:
    s = A[col].dropna()
    j = pd.concat([s.rename('w'), oni.rename('oni')], axis=1).dropna()
    j3 = j.rolling(3, center=True).mean().dropna()
    r, p = pearsonr(j.w, j.oni); rs, ps = spearmanr(j.w, j.oni)
    r3, p3 = pearsonr(j3.w, j3.oni)
    # annual
    ya = s.groupby(s.index.year).mean().where(s.groupby(s.index.year).count() >= 8)
    yo = oni.groupby(oni.index.year).mean()
    jy = pd.concat([ya.rename('w'), yo.rename('oni')], axis=1).dropna()
    ry, py = pearsonr(jy.w, jy.oni)
    res.append(dict(series=col, n_months=len(j), r_monthly=r, p_monthly=p, rho_monthly=rs, r_3m=r3, p_3m=p3, n_years=len(jy), r_annual=ry, p_annual=py))
R = pd.DataFrame(res)
R.to_csv(f'{OUT}/enso_corr.csv', index=False)

# annual anomalies table (% of station climatology) and ONI annual mean
Y = A.groupby(A.index.year).mean().where(A.groupby(A.index.year).count() >= 8)
Y['ONI'] = oni.groupby(oni.index.year).mean()
Y.round(1).to_csv(f'{OUT}/annual_anomalies.csv')
pd.set_option('display.width', 250)
print(R.round(3).to_string(index=False))
print(Y.round(1).to_string())
