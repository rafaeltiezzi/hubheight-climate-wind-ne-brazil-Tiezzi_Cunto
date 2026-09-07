"""Pipeline v2 - step 4: can ML regressors trained on the free-running Eta add skill beyond the seasonal cycle?

Monthly transfer functions obs_t = f(Eta_t, Eta_{t-1..3}, month) with leave-one-year-out CV, against
benchmarks (training climatology; monthly ratio bias correction). Also: an AR model with observed lags (one-step
skill only) and the 'compression' of the ML projections relative to the delta change (extrapolation limit).
Outputs: out/ml_cv.csv, out/ml_cv_summary.csv, out/ml_projection_compression.csv
"""
import pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')
OUT = 'pipe2/out'
P = pickle.load(open(f'{OUT}/prep.pkl', 'rb'))
obs, eta = P['obs'], P['eta']
ORDER = ['SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'LN1', 'LN2', 'LN3', 'LN4', 'LN5', 'JC1', 'JC2']
MEMBERS = ['BESM', 'CANESM2', 'HADGEM2-ES', 'MIROC5']
SCEN = ['RCP4.5', 'RCP8.5']
REF = (2006, 2024)


def models():
    return {'LR': LinearRegression(),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'RF': RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=0, n_jobs=1),
            'GB': GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=0)}


def features(em, idx):
    X = pd.DataFrame({'eta0': em.reindex(idx).values, 'eta1': em.shift(1).reindex(idx).values,
                      'eta2': em.shift(2).reindex(idx).values, 'eta3': em.shift(3).reindex(idx).values,
                      'sin': np.sin(2 * np.pi * idx.month / 12), 'cos': np.cos(2 * np.pi * idx.month / 12)}, index=idx)
    return X


def run_station(st):
    rows, comp = [], []
    o = obs[st]
    om = o['monthly'].dropna()
    years = sorted(set(om.index.year))
    for sc in SCEN:
        for mem in MEMBERS:
            e = eta.get((st, mem, sc))
            if e is None:
                continue
            em = e['monthly']
            X = features(em, om.index); y = om
            ok = X.notna().all(axis=1)
            X, y = X[ok], y[ok]
            # observed-lag AR features (one-step-ahead only)
            Xar = pd.DataFrame({'obs1': o['monthly'].shift(1).reindex(y.index).values, 'obs12': o['monthly'].shift(12).reindex(y.index).values,
                                'sin': X['sin'].values, 'cos': X['cos'].values}, index=y.index)
            preds = {k: pd.Series(index=y.index, dtype=float) for k in ['LR', 'KNN', 'RF', 'GB', 'CLIM', 'RATIO', 'AR-LR', 'AR-RF']}
            for yr in years:
                tr, te = y.index.year != yr, y.index.year == yr
                if te.sum() == 0 or tr.sum() < 24:
                    continue
                for k, mdl in models().items():
                    mdl.fit(X[tr], y[tr]); preds[k][te] = mdl.predict(X[te])
                clim = y[tr].groupby(y[tr].index.month).mean()
                preds['CLIM'][te] = clim.reindex(y.index[te].month).values
                ratio = (y[tr] / X['eta0'][tr]).groupby(y[tr].index.month).mean()
                preds['RATIO'][te] = X['eta0'][te].values * ratio.reindex(y.index[te].month).values
                okar = Xar.notna().all(axis=1)
                tr2, te2 = tr & okar, te & okar
                if te2.sum() > 0 and tr2.sum() > 24:
                    m1 = LinearRegression().fit(Xar[tr2], y[tr2]); preds['AR-LR'][te2] = m1.predict(Xar[te2])
                    m2 = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=0, n_jobs=1).fit(Xar[tr2], y[tr2])
                    preds['AR-RF'][te2] = m2.predict(Xar[te2])
            for k, p in preds.items():
                j = pd.concat([y.rename('y'), p.rename('p')], axis=1).dropna()
                anom_y = j.y - j.y.groupby(j.index.month).transform('mean')
                anom_p = j.p - j.p.groupby(j.index.month).transform('mean')
                rows.append(dict(station=st, paper=o['paper'], member=mem, scen=sc, model=k, n=len(j),
                                 mae=(j.p - j.y).abs().mean(), rmse=np.sqrt(((j.p - j.y) ** 2).mean()),
                                 mape=100 * ((j.p - j.y).abs() / j.y).mean(), r=np.corrcoef(j.p, j.y)[0, 1],
                                 r_anom=np.corrcoef(anom_p, anom_y)[0, 1] if anom_p.std() > 0 else np.nan))
            # projection compression: fit RF/LR on all years, apply to future Eta, compare with delta
            for k in ['RF', 'GB', 'LR']:
                mdl = models()[k].fit(X, y)
                base = mdl.predict(X).mean()
                for dname, (y0, y1) in {'2045-2054': (2045, 2054), '2070-2099': (2070, 2099)}.items():
                    idx = em.index[(em.index.year >= y0) & (em.index.year <= y1)]
                    Xf = features(em, idx).dropna()
                    pf = mdl.predict(Xf).mean()
                    ref = em[(em.index.year >= REF[0]) & (em.index.year <= REF[1])].mean()
                    fut = em.reindex(idx).mean()
                    comp.append(dict(station=st, paper=o['paper'], member=mem, scen=sc, model=k, decade=dname,
                                     dU_delta_pct=100 * (fut / ref - 1), dU_ml_pct=100 * (pf / base - 1)))
    print(st, 'done', flush=True)
    return rows, comp


if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(2) as pool:
        res = pool.map(run_station, ORDER)
    rows = [r for rr, cc in res for r in rr]; comp = [c for rr, cc in res for c in cc]

cv = pd.DataFrame(rows); cv.to_csv(f'{OUT}/ml_cv.csv', index=False)
summ = cv.groupby(['model']).agg(mae=('mae', 'mean'), rmse=('rmse', 'mean'), mape=('mape', 'mean'), r=('r', 'mean'), r_anom=('r_anom', 'mean')).round(3)
summ.to_csv(f'{OUT}/ml_cv_summary.csv')
summ_st = cv.groupby(['paper', 'model']).agg(mae=('mae', 'mean'), r_anom=('r_anom', 'mean')).round(3).unstack('model')
summ_st.to_csv(f'{OUT}/ml_cv_by_station.csv')
cp = pd.DataFrame(comp); cp.to_csv(f'{OUT}/ml_projection_compression.csv', index=False)
pd.set_option('display.width', 250)
print(summ)
print(summ_st)
for k in ['RF', 'GB', 'LR']:
    s = cp[cp.model == k]
    print(k, 'slope of ML change vs delta change:', np.polyfit(s.dU_delta_pct, s.dU_ml_pct, 1).round(3), 'r=', round(np.corrcoef(s.dU_delta_pct, s.dU_ml_pct)[0, 1], 3),
          'mean |delta|', round(s.dU_delta_pct.abs().mean(), 2), 'mean |ml|', round(s.dU_ml_pct.abs().mean(), 2))
