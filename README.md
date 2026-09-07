# Hub-height wind and climate change in Northeast Brazil — code and public data

Companion repository for the manuscript:

> Tiezzi, R.O., Cunto, G.A.B., Boniolo, V.R., Ferreira, I.E.P., Pereira, D.R.
> *Climate change and the wind resource of Northeast Brazil: what hub-height
> tower measurements and downscaled climate projections show.* Submitted to
> Applied Energy, 2026. (citation and DOI to be added upon publication)

## What is here

```
code/                      Analysis pipeline (Python 3)
  p2_prepare.py            hourly -> daily -> monthly -> annual chain, QC, shear
  p2_eval.py               evaluation of raw Eta W100 vs observations (Tables 3a-3b)
  p2_ml.py                 machine-learning transfer test, leave-one-year-out (Table 4)
  p2_proj.py               change factors + quantile delta mapping projections (Tables 5-6)
  p2_enso.py               ENSO (ONI) correlation analysis (Table S1)
  p2_atlantic.py           tropical Atlantic (TNA/TSA) correlation analysis (Table S2)
  p2_figs.py, p2_tables.py figures (PT/EN) and tables/spreadsheet
  read_eta_public.py       helper to read the Eta series shipped in data/
data/
  eta_w100_daily/          48 daily W100 series (2006-2099): 6 Eta 20-km grid cells
                           (A-F, see cells_index.csv) x 4 GCMs (BESM, CanESM2,
                           HadGEM2-ES, MIROC5) x 2 scenarios (RCP4.5, RCP8.5),
                           extracted from the PROJETA platform (CPTEC/INPE)
  indices/                 ONI v6 (NOAA CPC), TNA and TSA (NOAA PSL) monthly indices
```

## What is NOT here (and why)

The 12 meteorological-mast wind series (hourly/10-min, 78–95 m a.g.l.) are the
property of a private wind-farm operator and were provided under a
non-disclosure agreement that forbids distribution of the raw data in any form.
Observational results appear exclusively as the aggregated figures and tables of
the article. Scripts that ingest the confidential raw exports are therefore not
distributed; the analysis scripts document every processing rule (validity
thresholds, anomaly-based annual means, redundant-anemometer checks) so the
chain is fully auditable, and everything that depends only on public inputs
(Eta series, climate indices) is reproducible from this repository.

## Data sources and credits

- **Eta/PROJETA**: Eta regional model (20 km) nested in BESM, CanESM2,
  HadGEM2-ES and MIROC5, RCP4.5/RCP8.5 — Chou et al. (2014a,b). Data obtained
  from the PROJETA platform, CPTEC/INPE (https://projeta.cptec.inpe.br).
  The Eta-HadGEM2-ES member uses a 360-day calendar.
- **ONI v6**: NOAA Climate Prediction Center (ERSSTv6).
- **TNA/TSA**: NOAA Physical Sciences Laboratory (HadISST1.1, climatology
  1991–2020); Enfield et al. (1999).

## Requirements

Python ≥ 3.10 with: pandas, numpy, scipy, scikit-learn, pymannkendall,
matplotlib, openpyxl (see `requirements.txt`).

## License

Code: MIT (see LICENSE). The data files in `data/` are redistributed for
reproducibility with attribution and remain subject to the terms of their
original providers (CPTEC/INPE and NOAA).

## Contact

Rafael de Oliveira Tiezzi — rafaeltiezzi@ufscar.br
