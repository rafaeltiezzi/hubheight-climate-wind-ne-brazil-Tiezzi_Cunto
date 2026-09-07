"""Reads the public Eta W100 daily series shipped in this repository.

Each file data/eta_w100_daily/eta_{GCM}_{RCP}_cell{A-F}.csv.gz holds the daily
100-m wind speed (w100_ms) of one 20-km Eta grid cell (2006-2099), extracted from
the PROJETA platform (CPTEC/INPE). cells_index.csv maps cells to centre coordinates.
The Eta-HadGEM2-ES member uses a 360-day calendar (12 x 30-day months).
"""
import glob, os
import pandas as pd

def load(cell, gcm, rcp, base=os.path.join(os.path.dirname(__file__), "..", "data", "eta_w100_daily")):
    fn = os.path.join(base, f"eta_{gcm}_{rcp}_cell{cell}.csv.gz")
    return pd.read_csv(fn)

def load_all(base=os.path.join(os.path.dirname(__file__), "..", "data", "eta_w100_daily")):
    out = {}
    for fn in glob.glob(os.path.join(base, "eta_*.csv.gz")):
        name = os.path.basename(fn)[4:-7]
        gcm, rcp, cell = name.rsplit("_", 2)
        out[(cell.replace("cell", ""), gcm, rcp)] = pd.read_csv(fn)
    return out

if __name__ == "__main__":
    d = load("A", "CANESM2", "RCP85")
    print(d.head(), len(d))
