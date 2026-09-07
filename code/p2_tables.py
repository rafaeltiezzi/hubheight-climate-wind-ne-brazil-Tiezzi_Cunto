"""Pipeline v2 - step 6: manuscript tables (Markdown) + results workbook (xlsx). LANG = pt|en (decimal separator)."""
import pickle, sys, warnings
import numpy as np
import pandas as pd
import pymannkendall as mk

warnings.filterwarnings('ignore')
LANG = sys.argv[1] if len(sys.argv) > 1 else 'pt'
OUT = 'pipe2/out'
P = pickle.load(open(f'{OUT}/prep.pkl', 'rb'))
obs, eta = P['obs'], P['eta']
ev = pd.read_csv(f'{OUT}/eval_members.csv')
pm = pd.read_csv(f'{OUT}/proj_members.csv')
pe = pd.read_csv(f'{OUT}/proj_ensemble.csv')
cv = pd.read_csv(f'{OUT}/ml_cv.csv')
cp = pd.read_csv(f'{OUT}/ml_projection_compression.csv')
enso = pd.read_csv(f'{OUT}/enso_corr.csv')
ORDER = ['SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'LN1', 'LN2', 'LN3', 'LN4', 'LN5', 'JC1', 'JC2']
PAPER = {c: obs[c]['paper'] for c in ORDER}
MEMBERS = ['BESM', 'CANESM2', 'HADGEM2-ES', 'MIROC5']
MLAB = {'BESM': 'Eta-BESM', 'CANESM2': 'Eta-CanESM2', 'HADGEM2-ES': 'Eta-HadGEM2-ES', 'MIROC5': 'Eta-MIROC5'}
SCEN = ['RCP4.5', 'RCP8.5']
DEC = ['2025-2034', '2035-2044', '2045-2054']
CELLS = {'SR1': 'A', 'SR2': 'A', 'SR3': 'A', 'SR5': 'A', 'SR4': 'B', 'LN1': 'C', 'LN2': 'D', 'LN3': 'D', 'LN4': 'D', 'LN5': 'E', 'JC1': 'F', 'JC2': 'F'}


def f(x, d=1, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '–'
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace('.', ',') if LANG == 'pt' else s


def pval(p):
    if np.isnan(p): return '–'
    if p < 0.001: return '<0,001' if LANG == 'pt' else '<0.001'
    return f(p, 3)


def pettitt(x):
    """Pettitt (1979) change-point test; returns (index of change, p approx)."""
    x = np.asarray(x); n = len(x)
    r = pd.Series(x).rank().values
    U = np.array([2 * r[:k].sum() - k * (n + 1) for k in range(1, n)])
    k = int(np.argmax(np.abs(U))); K = abs(U[k])
    p = 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2))
    return k, min(p, 1.0)


md = []
# ---------------- Table 1: stations ----------------
rows = []
for st in ORDER:
    o = obs[st]; m = o['monthly']; d = o['daily']
    p0, p1 = o['period']
    comp = 100 * d.notna().sum() / len(d)
    e = eta.get((st, 'BESM', 'RCP4.5'))
    rows.append([o['paper'], o['uf'], f"{o['top_h']:g}", f"{p0:%m/%Y}–{p1:%m/%Y}", str(int(m.notna().sum())), f(comp, 0),
                 f(m.mean(), 2), f(o['alpha_mean'], 2), f(o['u100_monthly'].mean(), 2), CELLS[st],
                 f"{e['cell'][0]:.3f} / {e['cell'][1]:.3f}".replace('.', ',' if LANG == 'pt' else '.')])
hdr = (['Estação', 'UF', 'Altura (m)', 'Período', 'Meses válidos', 'Dias válidos (%)', 'Média (m/s)', 'α', 'U100 eq. (m/s)', 'Célula Eta', 'Centro da célula (lat / lon)']
       if LANG == 'pt' else ['Station', 'State', 'Height (m)', 'Period', 'Valid months', 'Valid days (%)', 'Mean (m/s)', 'α', 'U100 eq. (m/s)', 'Eta cell', 'Cell centre (lat / lon)'])
md.append('**Tabela 1.**\n\n| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows))

# ---------------- Table 2: trends ----------------
rows = []
trend_store = {}
for st in ORDER:
    o = obs[st]; m = o['monthly'].dropna(); a = o['annual'].dropna()
    anom = m - o['clim'].reindex(m.index.month).values
    r = mk.hamed_rao_modification_test(anom.values)
    ra = mk.original_test(a.values)
    k, pp = pettitt(a.values); cp_year = int(a.index[k])
    cv_ = 100 * a.std() / a.mean()
    rows.append([o['paper'], f(m.mean(), 2), f(cv_, 1), f(a.min(), 2) + f' ({int(a.idxmin())})', f(a.max(), 2) + f' ({int(a.idxmax())})',
                 f(r.slope * 120, 2, True), f(100 * r.slope * 120 / m.mean(), 1, True), f(r.z, 2), pval(r.p),
                 f(ra.slope * 10, 2, True), pval(ra.p), f'{cp_year}/{cp_year + 1}' + ('*' if pp < 0.05 else '')])
    trend_store[st] = dict(sen_m=r.slope * 120, sen_pct=100 * r.slope * 120 / m.mean(), p=r.p, z=r.z, cp=cp_year, cp_p=pp, cv=cv_, mean=m.mean())
hdr = (['Estação', 'Média (m/s)', 'CV anual (%)', 'Mín. anual (ano)', 'Máx. anual (ano)', 'Sen mensal (m/s/década)', 'Sen (%/década)', 'Z (MK-HR)', 'p (MK-HR)', 'Sen anual (m/s/década)', 'p (anual)', 'Ponto de mudança (Pettitt)']
       if LANG == 'pt' else ['Station', 'Mean (m/s)', 'Annual CV (%)', 'Annual min (year)', 'Annual max (year)', "Monthly Sen (m/s/decade)", 'Sen (%/decade)', 'Z (MK-HR)', 'p (MK-HR)', 'Annual Sen (m/s/decade)', 'p (annual)', 'Change point (Pettitt)'])
md.append('**Tabela 2.**\n\n| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows))
pd.DataFrame(trend_store).T.to_csv(f'{OUT}/table2_trends.csv')

# ---------------- Table 3a: bias per member (mean of the two scenarios) ----------------
rows = []
for st in ORDER:
    o = obs[st]; e = ev[(ev.station == st)]
    r = [o['paper'], f(e.obs_mean.iloc[0], 2), f(e.obs_mean_100.iloc[0], 2)]
    for mem in MEMBERS:
        s = e[e.member == mem]
        if len(s) == 0: r.append('–'); continue
        r.append(f(s.bias_pct.mean(), 1, True) + ' [' + f(s.bias_pct.min(), 1, True) + '; ' + f(s.bias_pct.max(), 1, True) + ']')
    ens = e[e.member.str.startswith('ENS')]
    r.append(f(ens.bias_pct.mean(), 1, True)); r.append(f(ens.bias100_pct.mean(), 1, True))
    rows.append(r)
hdr = (['Estação', 'Obs. (m/s)', 'Obs. U100 eq. (m/s)'] + [MLAB[m] + ' (%)' for m in MEMBERS] + ['Ensemble (%)', 'Ensemble vs U100 eq. (%)'])
md.append('**Tabela 3a.**\n\n| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows))

# ---------------- Table 3b: seasonal cycle / distribution metrics per member (mean over stations and scenarios) ----------------
rows = []
for mem in MEMBERS:
    s = ev[ev.member == mem]
    rows.append([MLAB[mem], str(s.station.nunique()), f(s.bias_pct.mean(), 1, True), f(s.rmse_monthly.mean(), 2), f(s.r_clim.mean(), 2),
                 f(s.amp_ratio.mean(), 2), f(s.r_anom.mean(), 2) + ' [' + f(s.r_anom.min(), 2) + '; ' + f(s.r_anom.max(), 2) + ']',
                 f(s.sd_anom_ratio.mean(), 2), f(s.pss_daily.mean(), 2), f(s.u3_ratio.mean(), 2)])
hdr = (['Membro', 'Estações', 'Viés médio (%)', 'RMSE mensal (m/s)', 'r ciclo sazonal', 'Razão de amplitude sazonal', 'r anomalias mensais [mín; máx]', 'Razão σ anomalias', 'PSS diário', 'Razão U³']
       if LANG == 'pt' else ['Member', 'Stations', 'Mean bias (%)', 'Monthly RMSE (m/s)', 'r seasonal cycle', 'Seasonal amplitude ratio', 'r monthly anomalies [min; max]', 'σ ratio anomalies', 'Daily PSS', 'U³ ratio'])
md.append('**Tabela 3b.**\n\n| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows))

# ---------------- Table 4: ML CV ----------------
lab = {'CLIM': 'Climatologia (treino)' if LANG == 'pt' else 'Climatology (training)', 'RATIO': 'Razão mensal Eta/obs' if LANG == 'pt' else 'Monthly Eta/obs ratio',
       'LR': 'Regressão linear (Eta)' if LANG == 'pt' else 'Linear regression (Eta)', 'KNN': 'KNN (Eta)', 'RF': 'Random Forest (Eta)', 'GB': 'Gradient Boosting (Eta)',
       'AR-LR': 'Regressão linear (defasagens obs.)' if LANG == 'pt' else 'Linear regression (obs. lags)', 'AR-RF': 'Random Forest (defasagens obs.)' if LANG == 'pt' else 'Random Forest (obs. lags)'}
rows = []
for k in ['CLIM', 'RATIO', 'LR', 'KNN', 'RF', 'GB', 'AR-LR', 'AR-RF']:
    s = cv[cv.model == k]
    ra = '–' if k == 'CLIM' else f(s.r_anom.mean(), 2)
    rows.append([lab[k], f(s.mae.mean(), 2), f(s.rmse.mean(), 2), f(s.mape.mean(), 1), f(s.r.mean(), 2), ra])
hdr = (['Modelo', 'MAE (m/s)', 'RMSE (m/s)', 'MAPE (%)', 'r', 'r das anomalias'] if LANG == 'pt' else ['Model', 'MAE (m/s)', 'RMSE (m/s)', 'MAPE (%)', 'r', 'r of anomalies'])
comp_txt = []
for k in ['LR', 'RF', 'GB']:
    s = cp[cp.model == k]; sl = np.polyfit(s.dU_delta_pct, s.dU_ml_pct, 1)[0]
    comp_txt.append(f"{lab[k]}: {f(sl, 2)}")
md.append('**Tabela 4.**\n\n| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows) +
          '\n\nCompressão do sinal (inclinação ΔU_ML vs ΔU_delta): ' + '; '.join(comp_txt))

# ---------------- Table 5: projections ----------------
rows = []
for st in ORDER:
    o = obs[st]
    for sc in SCEN:
        r = [o['paper'] if sc == 'RCP4.5' else '', sc]
        for dec in DEC:
            s = pe[(pe.station == st) & (pe.scen == sc) & (pe.decade == dec)].iloc[0]
            if s.n_members > 1:
                txt = f(s.dU_qdm_mean, 1, True) + ' [' + f(s.dU_qdm_min, 1, True) + '; ' + f(s.dU_qdm_max, 1, True) + ']'
                txt += f" ({max(s.n_pos, s.n_neg)}/{s.n_members})"
            else:
                txt = f(s.dU_qdm_mean, 1, True) + '†'
            r.append(txt)
        s = pe[(pe.station == st) & (pe.scen == sc) & (pe.decade == '2045-2054')].iloc[0]
        r.append(f(s.dU3_qdm_mean, 1, True) + (('' if s.n_members == 1 else ' [' + f(s.dU3_qdm_min, 1, True) + '; ' + f(s.dU3_qdm_max, 1, True) + ']')))
        r.append(f(s.dU_qdm_vs_recent_mean, 1, True))
        r.append(f(s.eta_trend_2025_2054_mean, 1, True) + f" ({s.n_trend_pos}/{s.n_members}; {s.n_trend_sig} sig.)")
        rows.append(r)
hdr = (['Estação', 'Cenário', 'ΔU 2025–2034 (%)', 'ΔU 2035–2044 (%)', 'ΔU 2045–2054 (%)', 'ΔP/A 2045–2054 (%)', 'ΔU 2045–2054 vs 2021–2025 (%)', 'Tendência Eta bruta 2025–2054 (%/década)']
       if LANG == 'pt' else ['Station', 'Scenario', 'ΔU 2025–2034 (%)', 'ΔU 2035–2044 (%)', 'ΔU 2045–2054 (%)', 'ΔP/A 2045–2054 (%)', 'ΔU 2045–2054 vs 2021–2025 (%)', 'Raw Eta trend 2025–2054 (%/decade)'])
md.append('**Tabela 5.**\n\n| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows))

# ---------------- Table 6: per-member changes 2045-2054 (QDM) ----------------
rows = []
for st in ORDER:
    o = obs[st]
    for sc in SCEN:
        r = [o['paper'] if sc == 'RCP4.5' else '', sc]
        for mem in MEMBERS:
            s = pm[(pm.station == st) & (pm.scen == sc) & (pm.member == mem) & (pm.decade == '2045-2054')]
            if len(s) == 0: r.append('–'); continue
            s = s.iloc[0]
            r.append(f(s.dU_qdm_pct, 1, True) + ('*' if s.mw_p < 0.05 else ''))
        s = pm[(pm.station == st) & (pm.scen == sc) & (pm.decade == '2070-2099')]
        r.append(f(s.dU_qdm_pct.mean(), 1, True) + ('' if len(s) == 1 else ' [' + f(s.dU_qdm_pct.min(), 1, True) + '; ' + f(s.dU_qdm_pct.max(), 1, True) + ']'))
        rows.append(r)
hdr = (['Estação', 'Cenário'] + [MLAB[m] for m in MEMBERS] + ['Ensemble 2070–2099 (%)'])
md.append('**Tabela 6.**\n\n| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows) +
          '\n\n\\* Mann–Whitney p < 0,05 (médias anuais da década vs 2006–2024 do próprio membro).')

# ---------------- Table S1: ENSO correlations ----------------
rows = []
for _, s in enso.iterrows():
    rows.append([s.series, str(int(s.n_months)), f(s.r_monthly, 2), pval(s.p_monthly), f(s.r_3m, 2), pval(s.p_3m), str(int(s.n_years)), f(s.r_annual, 2), pval(s.p_annual)])
hdr = ['Série', 'n (meses)', 'r mensal', 'p', 'r (média móvel 3 m)', 'p', 'n (anos)', 'r anual', 'p']
md.append('**Tabela S1.**\n\n| ' + ' | '.join(hdr) + ' |\n|' + '---|' * len(hdr) + '\n' + '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows))

open(f'{OUT}/tables_{LANG}.md', 'w').write('\n\n'.join(md))

# ---------------- workbook ----------------
with pd.ExcelWriter(f'{OUT}/Resultados_pipeline_v2.xlsx') as xw:
    inv = pd.DataFrame([dict(code=st, paper=obs[st]['paper'], uf=obs[st]['uf'], top_h=obs[st]['top_h'], low_h=obs[st]['low_h'], alpha=obs[st]['alpha_mean'],
                             start=obs[st]['period'][0], end=obs[st]['period'][1], months=int(obs[st]['monthly'].notna().sum()),
                             mean=obs[st]['monthly'].mean(), u100=obs[st]['u100_monthly'].mean(), eta_cell=CELLS[st]) for st in ORDER])
    inv.to_excel(xw, sheet_name='estacoes', index=False)
    pd.DataFrame(trend_store).T.to_excel(xw, sheet_name='tendencias')
    ev.to_excel(xw, sheet_name='avaliacao_eta', index=False)
    pm.to_excel(xw, sheet_name='projecoes_membros', index=False)
    pe.to_excel(xw, sheet_name='projecoes_ensemble', index=False)
    pd.read_csv(f'{OUT}/proj_monthly_cf.csv').to_excel(xw, sheet_name='fatores_mensais', index=False)
    cv.to_excel(xw, sheet_name='ml_validacao', index=False)
    cp.to_excel(xw, sheet_name='ml_compressao', index=False)
    enso.to_excel(xw, sheet_name='enso_correlacoes', index=False)
    pd.read_csv(f'{OUT}/annual_anomalies.csv', index_col=0).to_excel(xw, sheet_name='anomalias_anuais')
    ann = pd.DataFrame({obs[st]['paper']: obs[st]['annual'] for st in ORDER}); ann.to_excel(xw, sheet_name='medias_anuais_obs')
    mon = pd.DataFrame({obs[st]['paper']: obs[st]['monthly'] for st in ORDER}); mon.to_excel(xw, sheet_name='medias_mensais_obs')
print('tables written')
