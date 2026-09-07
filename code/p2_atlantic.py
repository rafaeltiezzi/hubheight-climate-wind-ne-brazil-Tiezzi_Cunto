"""ENSO and tropical Atlantic correlation analysis (Tables S1 and S2 of the paper).

Inputs:
  data/indices/oni_v6.csv          - ONI v6 (NOAA CPC, ERSSTv6), monthly
  data/indices/tna_hadisst.txt     - TNA index (NOAA PSL, HadISST1.1, climo 1991-2020)
  data/indices/tsa_hadisst.txt     - TSA index (idem)
  monthly_anomalies_pct.csv        - CONFIDENTIAL: monthly wind anomalies (%) per mast.
                                     Not distributed (non-disclosure agreement with the
                                     data owner); the aggregated results are Tables S1-S2.
Correlations: Pearson at monthly, 3-month rolling and annual scales (annual means
require >= 8 valid months); partial correlation of TSA controlling for ONI.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(__file__)
IDX = os.path.join(HERE, "..", "data", "indices")

def load_wide(path):
    rows = []
    for line in open(path):
        p = line.split()
        y = int(p[0])
        for m, v in enumerate(p[1:], 1):
            rows.append((pd.Timestamp(y, m, 1), float(v)))
    return pd.DataFrame(rows, columns=["date", "v"]).set_index("date")["v"]

def corr_sets(w, x):
    j = pd.concat([w.rename("w"), x.rename("x")], axis=1, sort=False).dropna()
    r_m, p_m = pearsonr(j.w, j.x)
    j3 = j.rolling(3, center=True).mean().dropna()
    r_3, p_3 = pearsonr(j3.w, j3.x)
    ya = w.groupby(w.index.year).mean().where(w.groupby(w.index.year).count() >= 8)
    yx = x.groupby(x.index.year).mean()
    jy = pd.concat([ya.rename("w"), yx.rename("x")], axis=1, sort=False).dropna()
    r_a, p_a = pearsonr(jy.w, jy.x)
    return dict(n=len(j), r_m=r_m, p_m=p_m, r_3=r_3, p_3=p_3, n_y=len(jy), r_a=r_a, p_a=p_a)

def partial_r(w, x, z):
    j = pd.concat([w.rename("w"), x.rename("x"), z.rename("z")], axis=1, sort=False).dropna()
    rwx = pearsonr(j.w, j.x)[0]; rwz = pearsonr(j.w, j.z)[0]; rxz = pearsonr(j.x, j.z)[0]
    return (rwx - rwz * rxz) / np.sqrt((1 - rwz**2) * (1 - rxz**2))

if __name__ == "__main__":
    oni = pd.read_csv(os.path.join(IDX, "oni_v6.csv"), index_col=0, parse_dates=True).iloc[:, 0]
    tna = load_wide(os.path.join(IDX, "tna_hadisst.txt"))
    tsa = load_wide(os.path.join(IDX, "tsa_hadisst.txt"))
    grad = tna - tsa
    anom_path = "monthly_anomalies_pct.csv"  # confidential input
    if not os.path.exists(anom_path):
        print("Confidential mast anomalies not available; indices loaded OK:",
              len(oni), len(tna), len(tsa))
        raise SystemExit
    A = pd.read_csv(anom_path, index_col=0, parse_dates=True)
    rows = []
    for col in A.columns:
        w = A[col].dropna()
        o = corr_sets(w, oni); s = corr_sets(w, tsa); t = corr_sets(w, tna); g = corr_sets(w, grad)
        rows.append(dict(series=col, n=o["n"], r_oni_m=o["r_m"], r_tsa_m=s["r_m"], p_tsa_m=s["p_m"],
                         r_tna_m=t["r_m"], r_grad_m=g["r_m"], r_tsa_a=s["r_a"], p_tsa_a=s["p_a"],
                         r_oni_a=o["r_a"], p_oni_a=o["p_a"], r_tsa_partial_oni=partial_r(w, tsa, oni)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
