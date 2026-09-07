"""Pipeline v2 - step 5: figures (300 dpi). LANG = 'pt' or 'en'."""
import pickle, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pymannkendall as mk

warnings.filterwarnings('ignore')
LANG = sys.argv[1] if len(sys.argv) > 1 else 'pt'
OUT = 'pipe2/out'
FIG = f'pipe2/fig_{LANG}'
import os; os.makedirs(FIG, exist_ok=True)
P = pickle.load(open(f'{OUT}/prep.pkl', 'rb'))
obs, eta = P['obs'], P['eta']
qdm = pickle.load(open(f'{OUT}/qdm_series.pkl', 'rb'))
pm = pd.read_csv(f'{OUT}/proj_members.csv')
pe = pd.read_csv(f'{OUT}/proj_ensemble.csv')
ev = pd.read_csv(f'{OUT}/eval_members.csv')
oni = pd.read_csv(f'{OUT}/oni.csv', index_col=0, parse_dates=True)['oni']
Y = pd.read_csv(f'{OUT}/annual_anomalies.csv', index_col=0)
ORDER = ['SR1', 'SR2', 'SR3', 'SR4', 'SR5', 'LN1', 'LN2', 'LN3', 'LN4', 'LN5', 'JC1', 'JC2']
PAPER = {c: obs[c]['paper'] for c in ORDER}
MEMBERS = ['BESM', 'CANESM2', 'HADGEM2-ES', 'MIROC5']
MLAB = {'BESM': 'Eta-BESM', 'CANESM2': 'Eta-CanESM2', 'HADGEM2-ES': 'Eta-HadGEM2-ES', 'MIROC5': 'Eta-MIROC5'}
MCOL = {'BESM': '#1b9e77', 'CANESM2': '#d95f02', 'HADGEM2-ES': '#7570b3', 'MIROC5': '#e7298a'}
SCEN = ['RCP4.5', 'RCP8.5']
SCOL = {'RCP4.5': '#1f77b4', 'RCP8.5': '#d62728'}
DEC = ['2025-2034', '2035-2044', '2045-2054']
T = {
    'pt': dict(ws='Velocidade do vento (m/s)', month='Mês', year='Ano', obs='Observado', annual='Média anual', trend='Tendência de Sen',
               months=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], clim='Climatologia mensal',
               pdf='Densidade de probabilidade', daily='Velocidade diária do vento (m/s)', raw='Eta bruto', ens='Média do ensemble',
               band='Faixa entre membros', dU='Variação da velocidade média (%)', dP='Variação da densidade de potência (%)',
               anom='Anomalia anual (%)', oni='ONI (°C)', ce='Ceará (5 estações)', rn='Rio Grande do Norte (7 estações)',
               elnino='El Niño', lanina='La Niña', proj='Projeção (QDM)', delta='Δ pelo fator de mudança (%)', ml='Δ projetado pelo regressor (%)',
               agree='≥3/4 membros com o mesmo sinal', obsann='Observado (média anual)', ref='Referência do modelo (2006–2024)'),
    'en': dict(ws='Wind speed (m/s)', month='Month', year='Year', obs='Observed', annual='Annual mean', trend="Sen's slope",
               months=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], clim='Monthly climatology',
               pdf='Probability density', daily='Daily wind speed (m/s)', raw='Raw Eta', ens='Ensemble mean',
               band='Member range', dU='Change in mean wind speed (%)', dP='Change in wind power density (%)',
               anom='Annual anomaly (%)', oni='ONI (°C)', ce='Ceará (5 stations)', rn='Rio Grande do Norte (7 stations)',
               elnino='El Niño', lanina='La Niña', proj='Projection (QDM)', delta='Δ from change factor (%)', ml='Δ projected by the regressor (%)',
               agree='≥3/4 members with the same sign', obsann='Observed (annual mean)', ref='Model reference (2006–2024)'),
}[LANG]
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 7, 'font.family': 'DejaVu Sans'})
LET = 'abcdefghijkl'


def enso_bands(ax, y0, y1):
    s = oni[(oni.index.year >= y0) & (oni.index.year <= y1)]
    for date, v in s.items():
        if v >= 0.5:
            ax.axvspan(date, date + pd.offsets.MonthBegin(1), color='#fbb4ae', alpha=0.35, lw=0)
        elif v <= -0.5:
            ax.axvspan(date, date + pd.offsets.MonthBegin(1), color='#b3cde3', alpha=0.45, lw=0)


# ---------- Fig 1: observed monthly series, annual means, Sen trend, ENSO bands ----------
fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.2), sharex=True)
for i, st in enumerate(ORDER):
    ax = axes.flat[i]; o = obs[st]
    m = o['monthly']; a = o['annual'].dropna()
    enso_bands(ax, 2011, 2025)
    ax.plot(m.index, m.values, color='0.55', lw=0.6)
    ax.plot(pd.to_datetime([f'{y}-07-01' for y in a.index]), a.values, 'o-', color='k', ms=2.5, lw=1)
    # Sen trend on deseasonalized monthly anomalies (Hamed-Rao MK), drawn through the mean of the series
    mm = m.dropna(); anom = mm - o['clim'].reindex(mm.index.month).values
    r = mk.hamed_rao_modification_test(anom.values)
    t = np.arange(len(mm)); yy = mm.mean() + r.slope * (t - t.mean())
    ax.plot(mm.index, yy, '--', color='#d62728', lw=1.2)
    ptxt = f"p={r.p:.2f}" if r.p >= 0.01 else "p<0,01" if LANG == 'pt' else "p<0.01"
    ax.text(0.02, 0.04, f"{100*r.slope*120/mm.mean():+.1f} %/" + ('década' if LANG == 'pt' else 'decade') + f" ({ptxt})", transform=ax.transAxes, fontsize=7, color='#d62728')
    ax.set_title(f"({LET[i]}) {PAPER[st]} – {o['uf']} – {o['top_h']:g} m", loc='left')
    ax.set_xlim(pd.Timestamp('2011-01-01'), pd.Timestamp('2026-01-01'))
    ax.set_xticks(pd.to_datetime([f'{y}-01-01' for y in range(2012, 2026, 3)]))
    ax.set_xticklabels([str(y) for y in range(2012, 2026, 3)])
    ax.grid(alpha=0.3)
    if i % 3 == 0: ax.set_ylabel(T['ws'])
for ax in axes[-1]: ax.set_xlabel(T['year'])
handles = [Line2D([], [], color='0.55', lw=0.8, label=T['clim'].split(' ')[0] + ' ' + ('mensal' if LANG == 'pt' else 'monthly')),
           Line2D([], [], color='k', marker='o', ms=3, lw=1, label=T['annual']),
           Line2D([], [], color='#d62728', ls='--', lw=1.2, label=T['trend']),
           Patch(color='#fbb4ae', alpha=0.6, label=T['elnino'] + ' (ONI ≥ 0,5)' if LANG == 'pt' else T['elnino'] + ' (ONI ≥ 0.5)'),
           Patch(color='#b3cde3', alpha=0.7, label=T['lanina'] + ' (ONI ≤ −0,5)' if LANG == 'pt' else T['lanina'] + ' (ONI ≤ −0.5)')]
handles[0].set_label('Média mensal' if LANG == 'pt' else 'Monthly mean')
fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f'{FIG}/fig1_observado.png', dpi=300); plt.close(fig)

# ---------- Fig 2: seasonal cycle obs vs Eta members ----------
fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.0), sharex=True)
for i, st in enumerate(ORDER):
    ax = axes.flat[i]; o = obs[st]
    om = o['monthly'].dropna(); oc = om.groupby(om.index.month).mean()
    ax.plot(range(1, 13), oc.values, 'k-o', ms=3, lw=1.6, label=T['obs'])
    for mem in MEMBERS:
        for sc in SCEN:
            e = eta.get((st, mem, sc))
            if e is None: continue
            em = e['monthly'].reindex(om.index); ec = em.groupby(em.index.month).mean()
            ax.plot(range(1, 13), ec.values, ls='-' if sc == 'RCP4.5' else '--', color=MCOL[mem], lw=1, alpha=0.9)
    ax.set_title(f"({LET[i]}) {PAPER[st]}", loc='left'); ax.set_xticks(range(1, 13)); ax.set_xticklabels(T['months'])
    ax.grid(alpha=0.3)
    if i % 3 == 0: ax.set_ylabel(T['ws'])
for ax in axes[-1]: ax.set_xlabel(T['month'])
handles = [Line2D([], [], color='k', marker='o', ms=3, lw=1.6, label=T['obs'])] + \
          [Line2D([], [], color=MCOL[m], lw=1.2, label=MLAB[m]) for m in MEMBERS] + \
          [Line2D([], [], color='0.3', lw=1, ls='-', label='RCP4.5'), Line2D([], [], color='0.3', lw=1, ls='--', label='RCP8.5')]
fig.legend(handles=handles, loc='lower center', ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f'{FIG}/fig2_ciclo_sazonal.png', dpi=300); plt.close(fig)

# ---------- Fig 3: daily PDFs obs vs raw Eta (RCP4.5 members) ----------
bins = np.arange(0, 18.5, 0.5)
fig, axes = plt.subplots(4, 3, figsize=(7.2, 7.6), sharex=True, sharey=True)
for i, st in enumerate(ORDER):
    ax = axes.flat[i]; o = obs[st]
    od = o['daily'].dropna(); om = o['monthly'].dropna()
    months = set(zip(om.index.year, om.index.month))
    od = od[[(y, m) in months for y, m in zip(od.index.year, od.index.month)]]
    ax.hist(od.values, bins=bins, density=True, color='0.75', label=T['obs'])
    for mem in MEMBERS:
        e = eta.get((st, mem, 'RCP4.5'))
        if e is None: continue
        d = e['daily']; d = d[[(y, m) in months for y, m in zip(d.year, d.month)]]['ws']
        h, _ = np.histogram(d.values, bins=bins, density=True)
        ax.plot(bins[:-1] + 0.25, h, color=MCOL[mem], lw=1.1)
    ax.set_title(f"({LET[i]}) {PAPER[st]}", loc='left'); ax.grid(alpha=0.3)
    if i % 3 == 0: ax.set_ylabel(T['pdf'])
for ax in axes[-1]: ax.set_xlabel(T['daily'])
handles = [Patch(color='0.75', label=T['obs'])] + [Line2D([], [], color=MCOL[m], lw=1.2, label=MLAB[m] + ' (RCP4.5)') for m in MEMBERS]
fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f'{FIG}/fig3_pdf_diaria.png', dpi=300); plt.close(fig)

# ---------- Fig 4: QDM-corrected annual series 2006-2054, ensemble per scenario, observed annual means ----------
fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.4), sharex=True)
for i, st in enumerate(ORDER):
    ax = axes.flat[i]; o = obs[st]
    for sc in SCEN:
        ser = [qdm[(st, mem, sc)] for mem in MEMBERS if (st, mem, sc) in qdm]
        if not ser: continue
        S = pd.concat(ser, axis=1)
        S10 = S.rolling(10, center=True, min_periods=6).mean()   # 10-yr running mean of each member
        ax.fill_between(S10.index, S10.min(axis=1), S10.max(axis=1), color=SCOL[sc], alpha=0.18, lw=0)
        ax.plot(S10.index, S10.mean(axis=1), color=SCOL[sc], lw=1.5, label=sc)
    a = o['annual'].dropna()
    ax.plot(a.index, a.values, 'ko', ms=2.8, label=T['obsann'])
    ax.axhline(a.mean(), color='k', lw=0.7, ls=':')
    ax.axvspan(2025, 2054, color='0.92', zorder=0)
    n = len([m for m in MEMBERS if (st, m, 'RCP4.5') in qdm])
    ax.set_title(f"({LET[i]}) {PAPER[st]} (n={n})", loc='left'); ax.grid(alpha=0.3); ax.set_xlim(2006, 2054)
    if i % 3 == 0: ax.set_ylabel(T['ws'])
for ax in axes[-1]: ax.set_xlabel(T['year'])
handles = [Line2D([], [], color=SCOL['RCP4.5'], lw=1.5, label='RCP4.5 – ' + T['ens']), Line2D([], [], color=SCOL['RCP8.5'], lw=1.5, label='RCP8.5 – ' + T['ens']),
           Patch(color='0.6', alpha=0.3, label=T['band'] + (' (média móvel de 10 anos)' if LANG == 'pt' else ' (10-yr running mean)')),
           Line2D([], [], color='k', marker='o', ls='', ms=3, label=T['obsann']),
           Line2D([], [], color='k', ls=':', lw=0.8, label=('Média observada' if LANG == 'pt' else 'Observed mean'))]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(f'{FIG}/fig4_series_projetadas.png', dpi=300); plt.close(fig)

# ---------- Fig 5: heatmaps of ensemble-mean changes (QDM) and member dots for 2045-2054 ----------
papers = [PAPER[s] for s in ORDER]
cols = [(d, sc) for sc in SCEN for d in DEC]
M = np.full((len(papers), len(cols)), np.nan); M3 = M.copy(); AG = np.zeros_like(M, dtype=bool); NM = np.zeros_like(M, dtype=int)
for i, p in enumerate(papers):
    for j, (d, sc) in enumerate(cols):
        r = pe[(pe.paper == p) & (pe.scen == sc) & (pe.decade == d)]
        if len(r):
            r = r.iloc[0]; M[i, j] = r.dU_qdm_mean; M3[i, j] = r.dU3_qdm_mean; NM[i, j] = r.n_members
            AG[i, j] = (max(r.n_pos, r.n_neg) >= 3) if r.n_members >= 3 else False
fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.2))
for ax, MM, title, vmax in [(axes[0], M, T['dU'], 8), (axes[1], M3, T['dP'], 24)]:
    im = ax.imshow(MM, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
    for i in range(MM.shape[0]):
        for j in range(MM.shape[1]):
            if np.isnan(MM[i, j]): continue
            txt = f"{MM[i, j]:+.1f}".replace('.', ',' if LANG == 'pt' else '.')
            if AG[i, j]: txt += '*'
            if NM[i, j] == 1: txt += '†'
            ax.text(j, i, txt, ha='center', va='center', fontsize=6.5, color='k')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([d for d, sc in cols], rotation=60, ha='right', fontsize=7)
    ax.set_yticks(range(len(papers))); ax.set_yticklabels(papers)
    ax.axvline(2.5, color='k', lw=0.8)
    ax.text(1, -0.8, 'RCP4.5', ha='center', fontsize=8); ax.text(4, -0.8, 'RCP8.5', ha='center', fontsize=8)
    ax.set_title(title, pad=18, fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=7)
axes[0].set_title('(a) ' + T['dU'], pad=18, fontsize=8); axes[1].set_title('(b) ' + T['dP'], pad=18, fontsize=8)
note = '* ' + T['agree'] + (('; † ' + ('apenas Eta-BESM disponível' if LANG == 'pt' else 'only Eta-BESM available')) if (NM == 1).any() else '')
fig.text(0.5, 0.01, note, ha='center', fontsize=7)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f'{FIG}/fig5_mapa_calor.png', dpi=300); plt.close(fig)

# ---------- Fig 6: member dots, 2045-2054 and 2070-2099 ----------
fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
for ax, dec in zip(axes, ['2045-2054', '2070-2099']):
    for i, p in enumerate(papers):
        for sc, dx in [('RCP4.5', -0.17), ('RCP8.5', 0.17)]:
            r = pm[(pm.paper == p) & (pm.scen == sc) & (pm.decade == dec)]
            for _, row in r.iterrows():
                ax.plot(i + dx, row.dU_qdm_pct, marker='o' if sc == 'RCP4.5' else 's', color=MCOL[row.member], ms=4, ls='', alpha=0.9, mec='k', mew=0.3)
            if len(r) >= 2:
                ax.plot([i + dx, i + dx], [r.dU_qdm_pct.min(), r.dU_qdm_pct.max()], color=SCOL[sc], lw=0.8, zorder=0)
                ax.plot(i + dx, r.dU_qdm_pct.mean(), marker='_', color=SCOL[sc], ms=10, mew=1.8)
    ax.axhline(0, color='k', lw=0.7); ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel(T['dU']); ax.set_title(('(a) ' if dec == '2045-2054' else '(b) ') + dec, loc='left')
axes[-1].set_xticks(range(len(papers))); axes[-1].set_xticklabels(papers)
handles = [Line2D([], [], color=MCOL[m], marker='o', ls='', ms=5, mec='k', mew=0.3, label=MLAB[m]) for m in MEMBERS] + \
          [Line2D([], [], color='0.4', marker='o', ls='', ms=5, label='RCP4.5'), Line2D([], [], color='0.4', marker='s', ls='', ms=5, label='RCP8.5'),
           Line2D([], [], color='0.4', marker='_', ls='', ms=10, mew=1.8, label=T['ens'])]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(f'{FIG}/fig6_membros.png', dpi=300); plt.close(fig)

# ---------- Fig 7: regional annual anomalies and ONI ----------
fig, ax = plt.subplots(figsize=(7.2, 3.0))
yrs = Y.index.values
w = 0.38
ax.bar(yrs - w / 2, Y['CE'].values, width=w, color='#1b9e77', label=T['ce'])
ax.bar(yrs + w / 2, Y['RN'].values, width=w, color='#d95f02', label=T['rn'])
ax.axhline(0, color='k', lw=0.7); ax.set_ylabel(T['anom']); ax.set_xlabel(T['year']); ax.grid(axis='y', alpha=0.3)
ax2 = ax.twinx(); ax2.plot(yrs, Y['ONI'].values, 'k-o', ms=3, lw=1.2, label=T['oni']); ax2.set_ylabel(T['oni']); ax2.set_ylim(-2.2, 2.2)
ax.set_ylim(-14, 14); ax.set_xticks(yrs)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper right', ncol=3, frameon=False)
fig.tight_layout(); fig.savefig(f'{FIG}/fig7_anomalias_enso.png', dpi=300); plt.close(fig)

print('figures written to', FIG)
