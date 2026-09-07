"""Pipeline v2 - step 2: projections per station x member x scenario x decade.

Methods:
  (a) change factor (delta) of the mean wind speed and of the mean cube (wind power density proxy),
      reference = model 2006-2024, future decades 2025-2034, 2035-2044, 2045-2054 (+ 2070-2099 for context);
  (b) quantile delta mapping (QDM, Cannon et al. 2015; multiplicative, by calendar month) of the daily
      series onto the observed daily distribution of each mast -> corrected future series -> decade means,
      mean cubes, annual means;
  (c) Mann-Whitney U test between annual means of each future decade and of the reference (raw model);
  (d) raw Eta trends 2025-2054 (Sen slope / Mann-Kendall on annual means).
Outputs: out/proj_members.csv, out/proj_ensemble.csv, out/proj_monthly_cf.csv, out/qdm_series.pkl
"""
import pickle, warnings
import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')
OUT = 'pipe2/out'
P = pickle.load(open(f'{OUT}/prep.pkl', 'rb'))
obs, eta = P['obs'], P['eta']
ORDER = ['SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'LN1', 'LN2', 'LN3', 'LN4', 'LN5', 'JC1', 'JC2']
MEMBERS = ['BESM', 'CANESM2', 'HADGEM2-ES', 'MIROC5']
SCEN = ['RCP4.5', 'RCP8.5']
REF = (2006, 2024)
DECADES = {'2025-2034': (2025, 2034), '2035-2044': (2035, 2044), '2045-2054': (2045, 2054), '2070-2099': (2070, 2099)}
TAUS = np.linspace(0.005, 0.995, 199)


def qdm_month(obs_vals, ref_vals, fut_vals):
    """Multiplicative quantile delta mapping for one calendar month (Cannon et al., 2015)."""
    q_obs = np.quantile(obs_vals, TAUS)
    q_ref = np.quantile(ref_vals, TAUS)
    fut_sorted = np.sort(fut_vals)
    # empirical non-exceedance probability of each future value within the future distribution
    tau = (np.searchsorted(fut_sorted, fut_vals, side='right') - 0.5) / len(fut_vals)
    tau = np.clip(tau, TAUS[0], TAUS[-1])
    q_fut_tau = np.interp(tau, TAUS, np.quantile(fut_vals, TAUS))
    q_ref_tau = np.interp(tau, TAUS, q_ref)
    q_obs_tau = np.interp(tau, TAUS, q_obs)
    delta = q_fut_tau / np.maximum(q_ref_tau, 0.1)
    return q_obs_tau * delta


def qm_month(obs_vals, ref_vals, vals):
    """Plain empirical quantile mapping (used for the reference period itself)."""
    q_obs = np.quantile(obs_vals, TAUS)
    q_ref = np.quantile(ref_vals, TAUS)
    ref_sorted = np.sort(ref_vals)
    tau = (np.searchsorted(ref_sorted, vals, side='right') - 0.5) / len(ref_vals)
    tau = np.clip(tau, TAUS[0], TAUS[-1])
    return np.interp(tau, TAUS, q_obs)


def sen_pct_dec(annual):
    a = annual.dropna()
    if len(a) < 8:
        return np.nan, np.nan
    r = mk.original_test(a.values)
    return 100 * r.slope * 10 / a.mean(), r.p


rows, cf_rows, qdm_series = [], [], {}
for st in ORDER:
    o = obs[st]
    od = o['daily'].dropna()
    od3 = o['daily3'].reindex(od.index)          # mean of hourly cubes per day
    obs_mean = od.mean()
    obs_u3 = (od ** 3).mean()                       # cube of daily means (like the model)
    obs_ann = o['annual'].dropna()
    obs_recent = obs_ann.loc[2021:2025].mean()
    obs_by_month = {m: od[od.index.month == m].values for m in range(1, 13)}
    for sc in SCEN:
        for mem in MEMBERS:
            e = eta.get((st, mem, sc))
            if e is None:
                continue
            d = e['daily']
            ref = d[(d.year >= REF[0]) & (d.year <= REF[1])]
            ref_by_month = {m: ref[ref.month == m]['ws'].values for m in range(1, 13)}
            ref_mean, ref_u3 = ref.ws.mean(), (ref.ws ** 3).mean()
            ann_raw = e['annual']
            ann_ref = ann_raw.loc[REF[0]:REF[1]]
            # QM of the reference period (for the corrected historical series)
            corr_ref = np.empty(len(ref));
            for m in range(1, 13):
                idx = (ref.month == m).values
                corr_ref[idx] = qm_month(obs_by_month[m], ref_by_month[m], ref[idx]['ws'].values)
            ref_c = ref.assign(wsc=corr_ref)
            ann_ref_c = ref_c.groupby('year')['wsc'].mean()
            corr_ref_mean, corr_ref_u3 = ref_c.wsc.mean(), (ref_c.wsc ** 3).mean()
            series_c = [ann_ref_c]
            t_2554, p_2554 = sen_pct_dec(ann_raw.loc[2025:2054])
            t_0654, p_0654 = sen_pct_dec(ann_raw.loc[2006:2054])
            for dname, (y0, y1) in DECADES.items():
                fut = d[(d.year >= y0) & (d.year <= y1)]
                fut_mean, fut_u3 = fut.ws.mean(), (fut.ws ** 3).mean()
                # monthly change factors
                for m in range(1, 13):
                    fm = fut[fut.month == m]['ws']; rm = ref_by_month[m]
                    cf_rows.append(dict(station=st, paper=o['paper'], member=mem, scen=sc, decade=dname, month=m,
                                        cf_u=fm.mean() / rm.mean(), cf_u3=(fm ** 3).mean() / (rm ** 3).mean(),
                                        obs_clim=o['clim'].get(m, np.nan)))
                # QDM
                corr = np.empty(len(fut))
                for m in range(1, 13):
                    idx = (fut.month == m).values
                    corr[idx] = qdm_month(obs_by_month[m], ref_by_month[m], fut[idx]['ws'].values)
                fut_c = fut.assign(wsc=corr)
                ann_fut_c = fut_c.groupby('year')['wsc'].mean()
                if dname != '2070-2099':
                    series_c.append(ann_fut_c)
                qdm_mean, qdm_u3 = fut_c.wsc.mean(), (fut_c.wsc ** 3).mean()
                # Mann-Whitney on annual means (raw model): future decade vs reference
                mw = mannwhitneyu(ann_raw.loc[y0:y1].values, ann_ref.values, alternative='two-sided')
                rows.append(dict(
                    station=st, paper=o['paper'], uf=o['uf'], member=mem, scen=sc, decade=dname,
                    obs_mean=obs_mean, obs_recent_2021_2025=obs_recent,
                    ref_mean_raw=ref_mean, fut_mean_raw=fut_mean,
                    dU_delta_pct=100 * (fut_mean / ref_mean - 1),
                    dU3_delta_pct=100 * (fut_u3 / ref_u3 - 1),
                    proj_mean_delta=obs_mean * fut_mean / ref_mean,
                    qdm_ref_mean=corr_ref_mean, qdm_fut_mean=qdm_mean,
                    dU_qdm_pct=100 * (qdm_mean / corr_ref_mean - 1),
                    dU3_qdm_pct=100 * (qdm_u3 / corr_ref_u3 - 1),
                    dU_qdm_vs_obs_pct=100 * (qdm_mean / obs_mean - 1),
                    dU_qdm_vs_recent_pct=100 * (qdm_mean / obs_recent - 1),
                    mw_p=mw.pvalue, sd_ann_ref=ann_ref.std(), sd_ann_fut=ann_raw.loc[y0:y1].std(),
                    eta_trend_2025_2054_pct_dec=t_2554, eta_trend_2025_2054_p=p_2554,
                    eta_trend_2006_2054_pct_dec=t_0654, eta_trend_2006_2054_p=p_0654))
            qdm_series[(st, mem, sc)] = pd.concat(series_c).sort_index()

pm = pd.DataFrame(rows)
pm.to_csv(f'{OUT}/proj_members.csv', index=False)
pd.DataFrame(cf_rows).to_csv(f'{OUT}/proj_monthly_cf.csv', index=False)
pickle.dump(qdm_series, open(f'{OUT}/qdm_series.pkl', 'wb'))

# ensemble summaries per station x scenario x decade (and all members x scenarios pooled)
ens_rows = []
for (st, sc, dec), g in pm.groupby(['station', 'scen', 'decade']):
    n = len(g)
    ens_rows.append(dict(station=st, paper=g.paper.iloc[0], uf=g.uf.iloc[0], scen=sc, decade=dec, n_members=n,
                         obs_mean=g.obs_mean.iloc[0],
                         dU_delta_mean=g.dU_delta_pct.mean(), dU_delta_min=g.dU_delta_pct.min(), dU_delta_max=g.dU_delta_pct.max(),
                         dU_qdm_mean=g.dU_qdm_pct.mean(), dU_qdm_min=g.dU_qdm_pct.min(), dU_qdm_max=g.dU_qdm_pct.max(),
                         dU3_delta_mean=g.dU3_delta_pct.mean(), dU3_delta_min=g.dU3_delta_pct.min(), dU3_delta_max=g.dU3_delta_pct.max(),
                         dU3_qdm_mean=g.dU3_qdm_pct.mean(), dU3_qdm_min=g.dU3_qdm_pct.min(), dU3_qdm_max=g.dU3_qdm_pct.max(),
                         n_pos=int((g.dU_delta_pct > 0).sum()), n_neg=int((g.dU_delta_pct < 0).sum()),
                         n_sig=int((g.mw_p < 0.05).sum()), n_sig_pos=int(((g.mw_p < 0.05) & (g.dU_delta_pct > 0)).sum()),
                         n_sig_neg=int(((g.mw_p < 0.05) & (g.dU_delta_pct < 0)).sum()),
                         dU_qdm_vs_recent_mean=g.dU_qdm_vs_recent_pct.mean(),
                         proj_mean_qdm=g.qdm_fut_mean.mean(),
                         eta_trend_2025_2054_mean=g.eta_trend_2025_2054_pct_dec.mean(),
                         n_trend_pos=int((g.eta_trend_2025_2054_pct_dec > 0).sum()),
                         n_trend_sig=int((g.eta_trend_2025_2054_p < 0.05).sum())))
for (st, dec), g in pm.groupby(['station', 'decade']):
    ens_rows.append(dict(station=st, paper=g.paper.iloc[0], uf=g.uf.iloc[0], scen='ALL', decade=dec, n_members=len(g),
                         obs_mean=g.obs_mean.iloc[0],
                         dU_delta_mean=g.dU_delta_pct.mean(), dU_delta_min=g.dU_delta_pct.min(), dU_delta_max=g.dU_delta_pct.max(),
                         dU_qdm_mean=g.dU_qdm_pct.mean(), dU_qdm_min=g.dU_qdm_pct.min(), dU_qdm_max=g.dU_qdm_pct.max(),
                         dU3_delta_mean=g.dU3_delta_pct.mean(), dU3_delta_min=g.dU3_delta_pct.min(), dU3_delta_max=g.dU3_delta_pct.max(),
                         dU3_qdm_mean=g.dU3_qdm_pct.mean(), dU3_qdm_min=g.dU3_qdm_pct.min(), dU3_qdm_max=g.dU3_qdm_pct.max(),
                         n_pos=int((g.dU_delta_pct > 0).sum()), n_neg=int((g.dU_delta_pct < 0).sum()),
                         n_sig=int((g.mw_p < 0.05).sum()), n_sig_pos=int(((g.mw_p < 0.05) & (g.dU_delta_pct > 0)).sum()),
                         n_sig_neg=int(((g.mw_p < 0.05) & (g.dU_delta_pct < 0)).sum()),
                         dU_qdm_vs_recent_mean=g.dU_qdm_vs_recent_pct.mean(), proj_mean_qdm=g.qdm_fut_mean.mean(),
                         eta_trend_2025_2054_mean=g.eta_trend_2025_2054_pct_dec.mean(),
                         n_trend_pos=int((g.eta_trend_2025_2054_pct_dec > 0).sum()),
                         n_trend_sig=int((g.eta_trend_2025_2054_p < 0.05).sum())))
pe = pd.DataFrame(ens_rows)
pe.to_csv(f'{OUT}/proj_ensemble.csv', index=False)

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
print(pm[['paper', 'member', 'scen', 'decade', 'obs_mean', 'ref_mean_raw', 'fut_mean_raw', 'dU_delta_pct', 'dU_qdm_pct', 'dU3_delta_pct', 'dU3_qdm_pct',
          'qdm_fut_mean', 'dU_qdm_vs_recent_pct', 'mw_p', 'eta_trend_2025_2054_pct_dec', 'eta_trend_2025_2054_p']].round(2).to_string(index=False))
print()
print(pe[['paper', 'scen', 'decade', 'n_members', 'dU_delta_mean', 'dU_delta_min', 'dU_delta_max', 'dU_qdm_mean', 'dU3_qdm_mean', 'n_pos', 'n_neg', 'n_sig',
          'dU_qdm_vs_recent_mean', 'eta_trend_2025_2054_mean', 'n_trend_pos', 'n_trend_sig']].round(1).to_string(index=False))
