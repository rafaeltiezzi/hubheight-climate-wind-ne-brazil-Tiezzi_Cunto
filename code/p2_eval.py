"""Pipeline v2 - step 1: evaluation of the raw Eta W100 against the mast observations (station period).

Outputs: out/eval_members.csv (per station x member x scenario), out/eval_summary.csv (per station, ensemble).
"""
import pickle, warnings
import numpy as np
import pandas as pd
import pymannkendall as mk

warnings.filterwarnings('ignore')
OUT = 'pipe2/out'
P = pickle.load(open(f'{OUT}/prep.pkl', 'rb'))
obs, eta = P['obs'], P['eta']
ORDER = ['SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'LN1', 'LN2', 'LN3', 'LN4', 'LN5', 'JC1', 'JC2']
MEMBERS = ['BESM', 'CANESM2', 'HADGEM2-ES', 'MIROC5']
SCEN = ['RCP4.5', 'RCP8.5']
BINS = np.arange(0, 20.5, 0.5)


def pss(a, b):
    ha, _ = np.histogram(a, bins=BINS, density=True)
    hb, _ = np.histogram(b, bins=BINS, density=True)
    w = np.diff(BINS)
    return float(np.sum(np.minimum(ha * w, hb * w)))


def sen_pct_dec(annual):
    a = annual.dropna()
    if len(a) < 8:
        return np.nan, np.nan
    r = mk.original_test(a.values)
    return 100 * r.slope * 10 / a.mean(), r.p


def eta_daily_series(e):
    """Daily Eta values keyed by (year, month) for PDF comparison (calendar-agnostic)."""
    d = e['daily']
    return d


rows = []
for st in ORDER:
    o = obs[st]
    om = o['monthly'].dropna()
    om100 = o['u100_monthly'].reindex(om.index)
    o_clim = om.groupby(om.index.month).mean()
    o_anom = om - o_clim.reindex(om.index.month).values
    od = o['daily'].dropna()
    od = od[(od.index >= om.index.min()) & (od.index <= om.index.max() + pd.offsets.MonthEnd(0))]
    obs_months = set(zip(om.index.year, om.index.month))
    od = od[[(y, m) in obs_months for y, m in zip(od.index.year, od.index.month)]]
    ens = {sc: [] for sc in SCEN}
    for sc in SCEN:
        for mem in MEMBERS:
            e = eta.get((st, mem, sc))
            if e is None:
                continue
            em = e['monthly'].reindex(om.index)
            ok = em.notna()
            em, om_, om100_ = em[ok], om[ok], om100[ok]
            e_clim = em.groupby(em.index.month).mean()
            e_anom = em - e_clim.reindex(em.index.month).values
            o_anom_ = o_anom[ok]
            ed = e['daily']
            ed = ed[[(y, m) in obs_months for y, m in zip(ed['year'], ed['month'])]]['ws']
            # trends of the model in the station period and in 2006-2025
            ann = e['annual']
            yrs = sorted(set(om.index.year))
            t_period, p_period = sen_pct_dec(ann.reindex(yrs))
            t_0625, p_0625 = sen_pct_dec(ann.loc[2006:2025])
            rows.append(dict(
                station=st, paper=o['paper'], uf=o['uf'], member=mem, scen=sc, cell_lat=e['cell'][0], cell_lon=e['cell'][1],
                n_months=int(ok.sum()), obs_mean=om_.mean(), obs_mean_100=om100_.mean(), eta_mean=em.mean(),
                bias=em.mean() - om_.mean(), bias_pct=100 * (em.mean() / om_.mean() - 1),
                bias100_pct=100 * (em.mean() / om100_.mean() - 1),
                rmse_monthly=float(np.sqrt(((em - om_) ** 2).mean())),
                r_monthly=float(np.corrcoef(em, om_)[0, 1]),
                r_clim=float(np.corrcoef(e_clim.reindex(o_clim.index), o_clim)[0, 1]),
                amp_obs=float(o_clim.max() - o_clim.min()), amp_eta=float(e_clim.max() - e_clim.min()),
                amp_ratio=float((e_clim.max() - e_clim.min()) / (o_clim.max() - o_clim.min())),
                month_max_obs=int(o_clim.idxmax()), month_max_eta=int(e_clim.idxmax()),
                month_min_obs=int(o_clim.idxmin()), month_min_eta=int(e_clim.idxmin()),
                r_anom=float(np.corrcoef(e_anom, o_anom_)[0, 1]),
                sd_anom_obs=float(o_anom_.std()), sd_anom_eta=float(e_anom.std()), sd_anom_ratio=float(e_anom.std() / o_anom_.std()),
                pss_daily=pss(od.values, ed.values),
                u3_ratio=float((ed ** 3).mean() / (od ** 3).mean()),
                sd_daily_obs=float(od.std()), sd_daily_eta=float(ed.std()),
                eta_trend_period_pct_dec=t_period, eta_trend_period_p=p_period,
                eta_trend_0625_pct_dec=t_0625, eta_trend_0625_p=p_0625,
                date_source=e['date_source']))
            ens[sc].append(em)
    # ensemble mean (members available)
    for sc in SCEN:
        if not ens[sc]:
            continue
        E = pd.concat(ens[sc], axis=1).mean(axis=1)
        ok = E.notna()
        E, om_, om100_ = E[ok], om[ok], om100[ok]
        e_clim = E.groupby(E.index.month).mean()
        rows.append(dict(station=st, paper=o['paper'], uf=o['uf'], member=f'ENS{len(ens[sc])}', scen=sc,
                         n_months=int(ok.sum()), obs_mean=om_.mean(), obs_mean_100=om100_.mean(), eta_mean=E.mean(),
                         bias=E.mean() - om_.mean(), bias_pct=100 * (E.mean() / om_.mean() - 1), bias100_pct=100 * (E.mean() / om100_.mean() - 1),
                         rmse_monthly=float(np.sqrt(((E - om_) ** 2).mean())), r_monthly=float(np.corrcoef(E, om_)[0, 1]),
                         r_clim=float(np.corrcoef(e_clim.reindex(o_clim.index), o_clim)[0, 1]),
                         amp_obs=float(o_clim.max() - o_clim.min()), amp_eta=float(e_clim.max() - e_clim.min()),
                         amp_ratio=float((e_clim.max() - e_clim.min()) / (o_clim.max() - o_clim.min())),
                         month_max_obs=int(o_clim.idxmax()), month_max_eta=int(e_clim.idxmax()),
                         month_min_obs=int(o_clim.idxmin()), month_min_eta=int(e_clim.idxmin())))

ev = pd.DataFrame(rows)
ev.to_csv(f'{OUT}/eval_members.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 40)
cols = ['paper', 'member', 'scen', 'n_months', 'obs_mean', 'obs_mean_100', 'eta_mean', 'bias_pct', 'bias100_pct', 'rmse_monthly', 'r_clim', 'amp_ratio',
        'month_max_obs', 'month_max_eta', 'r_anom', 'sd_anom_ratio', 'pss_daily', 'u3_ratio', 'eta_trend_period_pct_dec', 'eta_trend_0625_pct_dec']
print(ev[cols].round(2).to_string(index=False))
