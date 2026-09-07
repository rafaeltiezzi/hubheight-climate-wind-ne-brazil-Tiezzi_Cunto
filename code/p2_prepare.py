"""Pipeline v2 - step 0: prepare observed and Eta series (proper daily/monthly/annual means).

Outputs (pickle): pipe2/out/prep.pkl with
  obs[code] = dict(meta, hourly (all levels), daily, monthly, annual, clim, alpha, u100 monthly)
  eta[(code, member, scen)] = dict(daily DataFrame(year, month, day, ws), monthly Series, annual Series, cell)
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, 'icdata')
from load_obs import STATIONS, read_export, BASE
from eta_load import load_all, STATION_DIR

warnings.filterwarnings('ignore')
OUT = 'pipe2/out'
MIN_HOURS = 18      # hours per valid day
MIN_DAYS = 20       # valid days per valid month
MIN_MONTHS = 8      # valid months per valid year (annual mean via anomalies; 2015 in CE has 9 months)

ORDER = ['SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'LN1', 'LN2', 'LN3', 'LN4', 'LN5', 'JC1', 'JC2']
PAPER = {c: STATIONS[c][0] for c in ORDER}


def load_station_all_levels(code):
    paper, cplx, uf, files = STATIONS[code]
    parts, metas = [], []
    for f in files:
        meta, df = read_export(os.path.join(BASE, f))
        meta['file'] = f
        metas.append(meta); parts.append(df)
    df = pd.concat(parts, ignore_index=True).sort_values('dt')
    df['hour'] = df['dt'].dt.floor('h')
    cols = [c for c in df.columns if c.startswith('ws') and c != 'ws_top']
    hourly = df.groupby('hour')[cols + ['ws_top']].mean()
    hourly['n_sub'] = df.groupby('hour')['ws_top'].count()
    return metas, hourly


def annual_from_monthly(m, clim, min_months=MIN_MONTHS):
    """Annual mean from monthly means using anomalies relative to the climatology (robust to missing months)."""
    anom = m - clim.reindex(m.index.month).values
    g = anom.groupby(m.index.year)
    a = g.mean().where(g.count() >= min_months)
    return a + clim.mean()


def prepare_obs():
    obs = {}
    for code in ORDER:
        metas, hourly = load_station_all_levels(code)
        top = metas[0]['top_h']
        # daily means (>= 18 hourly values)
        g = hourly['ws_top'].groupby(hourly.index.date)
        daily = g.mean().where(g.count() >= MIN_HOURS)
        daily.index = pd.to_datetime(daily.index)
        daily = daily.asfreq('D')
        d3 = (hourly['ws_top'] ** 3).groupby(hourly.index.date).mean()   # mean of hourly cubes (for WPD)
        d3.index = pd.to_datetime(d3.index); d3 = d3.where(g.count() >= MIN_HOURS).asfreq('D')
        # monthly
        gm = daily.groupby([daily.index.year, daily.index.month])
        monthly = gm.mean().where(gm.count() >= MIN_DAYS)
        monthly.index = pd.to_datetime([f'{y}-{m:02d}-01' for y, m in monthly.index])
        monthly = monthly.asfreq('MS')
        clim = monthly.groupby(monthly.index.month).mean()
        annual = annual_from_monthly(monthly, clim)
        # shear: two distinct levels (top and the lowest level), monthly mean speeds
        low = float(metas[0]['heights'][0])   # lowest level of the main (historical) mast configuration
        if code == 'JC1':
            low = 30.0   # the '58 m' column of the historical JC1 export is inconsistent (reads ~27% below 80 m); use the 2024-25 30/80 m pair
        # consistency of the two redundant top-level anemometers (annual ratio)
        tops = sorted([float(c[2:]) for c in hourly.columns if c.startswith('ws') and c != 'ws_top'])[-2:]
        ratio_top = None
        if len(tops) == 2:
            a, b = hourly[f'ws{tops[0]:g}'], hourly[f'ws{tops[1]:g}']
            ok2 = a.notna() & b.notna()
            rr = (a[ok2] / b[ok2]).groupby(a[ok2].index.year).mean()
            ratio_top = rr.round(3).to_dict()
        hl, ht = hourly[f'ws{low:g}'], hourly['ws_top']
        ok = hl.notna() & ht.notna() & (hl > 1) & (ht > 1)
        alpha_h = np.log(ht[ok] / hl[ok]) / np.log(top / low)
        alpha_m = alpha_h.groupby([alpha_h.index.year, alpha_h.index.month]).mean()
        alpha_m.index = pd.to_datetime([f'{y}-{m:02d}-01' for y, m in alpha_m.index])
        alpha_clim = alpha_m.groupby(alpha_m.index.month).mean()
        alpha_mean = float(alpha_h.mean())
        # 100 m equivalent (power law with monthly climatological alpha)
        f100 = (100.0 / top) ** alpha_clim.reindex(monthly.index.month).values
        u100_monthly = monthly * f100
        obs[code] = dict(paper=PAPER[code], uf=STATIONS[code][2], complex=STATIONS[code][1], metas=metas, top_h=top, low_h=low,
                         hourly=hourly, daily=daily, daily3=d3, monthly=monthly, clim=clim, annual=annual,
                         alpha_mean=alpha_mean, alpha_clim=alpha_clim, u100_monthly=u100_monthly, ratio_top=ratio_top,
                         period=(monthly.dropna().index.min(), monthly.dropna().index.max()))
        print(code, PAPER[code], 'top', top, 'low', low, 'alpha', round(alpha_mean, 3), 'months', int(monthly.notna().sum()),
              'years', int(annual.notna().sum()), 'mean', round(monthly.mean(), 2), 'u100', round(u100_monthly.mean(), 2), 'ratio_top', ratio_top)
    return obs


def prepare_eta():
    data = load_all()
    eta = {}
    for (st, mem, sc), df in data.items():
        gm = df.groupby(['year', 'month'])['ws']
        monthly = gm.mean()
        monthly.index = pd.to_datetime([f'{y}-{m:02d}-01' for y, m in monthly.index])
        monthly = monthly.asfreq('MS')
        m3 = (df['ws'] ** 3).groupby([df['year'], df['month']]).mean()
        m3.index = monthly.index[:len(m3)] if len(m3) == len(monthly) else pd.to_datetime([f'{y}-{m:02d}-01' for y, m in m3.index])
        annual = df.groupby('year')['ws'].mean()
        annual3 = (df['ws'] ** 3).groupby(df['year']).mean()
        eta[(st, mem, sc)] = dict(daily=df, monthly=monthly, monthly3=m3, annual=annual, annual3=annual3,
                                  cell=(df.attrs['lat'], df.attrs['lng']), date_source=df.attrs['date_source'])
    return eta


if __name__ == '__main__':
    obs = prepare_obs()
    eta = prepare_eta()
    # report months missing in HadGEM RCP4.5
    for k, v in eta.items():
        if k[1] == 'HADGEM2-ES' and k[0] == 'JC1':
            cnt = v['daily'].groupby(['year', 'month']).size()
            print(k, 'months with <30 days:', cnt[cnt < 30].to_dict(), 'n months', len(cnt))
    pickle.dump(dict(obs=obs, eta=eta), open(f'{OUT}/prep.pkl', 'wb'))
    print('saved')
