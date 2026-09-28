"""EDS-2026-0083 revision: Protocol B interval definition check.

The manuscript and the response letter describe the Protocol B interval as the min-max
envelope of the four members; the code (evaluate_protocol_b.coverage_row,
conformal.evaluate_cell_conformal) uses the 2.5-97.5 percentiles of the four members
with linear interpolation, which lie 7.5 % of the outer member gaps inside the envelope.
Per condition, uncalibrated and fixed-map Platt: coverage under both definitions, mean
widths, silent counts (passes the 0.70 gate after contamination, coverage < 0.90) under
both, and the mean |consensus - benchmark| next to the width by track and level.
Output: reports/rev_eds/interval_definition_check.csv.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
MASTER = Path(os.environ.get("TS_MASTER_CSV") or
              REPO.parent / "sdm-robustness/data/raw/combined_data_true_master.csv")
os.environ.setdefault("TS_MASTER_CSV", str(MASTER))
SURF = REPO / "data/replicate_surfaces_protocol_b"
PLATT = REPO / "data/rev_eds/protocol_b_platt"
OUT = REPO / "reports/rev_eds"
REPS = ["rep_00", "rep_01", "rep_02", "rep_03"]
THRESH, GATE = 0.90, 0.70
KEYS = ["method", "entity_dir", "track", "level"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def members(root, cell):
    return pd.concat([pd.read_parquet(root / cell / f"{r}.parquet").set_index("subc_id")["predicted_probability"]
                      for r in REPS], axis=1)


def main():
    rpb = load_module("run_protocol_b", REPO / "scripts/run_protocol_b.py")
    from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR as E2D

    rows = []
    for method, root in (("uncalibrated", SURF), ("platt", PLATT)):
        for e in rpb.ENTITIES:
            edir = E2D[e]
            for t in rpb.TRACKS:
                B = members(root, f"{edir}__consensus__{t}__benchmark__L0").mean(axis=1)
                for lvl in rpb.LEVELS:
                    M = members(root, f"{edir}__consensus__{t}__lowacc__L{lvl}")
                    idx = M.index.intersection(B.index)
                    m = M.loc[idx].to_numpy()
                    b = B.loc[idx].to_numpy()
                    lo, hi = np.percentile(m, [2.5, 97.5], axis=1)
                    mn, mx = m.min(axis=1), m.max(axis=1)
                    rows.append({"method": method, "entity_dir": edir, "track": t, "level": lvl,
                                 "coverage_pct": float(np.mean((b >= lo) & (b <= hi))),
                                 "coverage_minmax": float(np.mean((b >= mn) & (b <= mx))),
                                 "width_pct": float(np.mean(hi - lo)), "width_minmax": float(np.mean(mx - mn)),
                                 "mean_abs_div": float(np.mean(np.abs(m.mean(axis=1) - b)))})
    df = pd.DataFrame(rows)
    pan = pd.read_csv(OUT / "protocol_b_platt_panel.csv")
    g = df.merge(pan[KEYS + ["coverage"]], on=KEYS)
    print(f"gate: percentile coverage vs protocol_b_platt_panel.csv, {len(g)} conditions, "
          f"max |diff| {(g['coverage_pct'] - g['coverage']).abs().max():.2e}")
    oos = pd.read_csv(OUT / "protocol_b_platt_oos.csv")
    df = df.merge(oos[oos["level"] > 0][KEYS + ["auc"]], on=KEYS, how="left")
    if df["auc"].isna().any():
        print(f"warning: no contaminated AUC for {int(df['auc'].isna().sum())} conditions")
    df.round(5).to_csv(OUT / "interval_definition_check.csv", index=False)

    print("\nmean coverage over units, percentile interval vs min-max envelope:")
    print(df.groupby(["method", "track", "level"])[["coverage_pct", "coverage_minmax"]].mean().round(3).to_string())
    print(f"largest per-condition gain of min-max over percentile: {(df['coverage_minmax'] - df['coverage_pct']).max():.3f}; "
          f"width ratio percentile / min-max, median {(df['width_pct'] / df['width_minmax']).median():.3f}")
    for method in ("uncalibrated", "platt"):
        d = df[df["method"] == method]
        for col in ("coverage_pct", "coverage_minmax"):
            s = [f"{int(((d['level'] == l) & (d['auc'] >= GATE) & (d[col] < THRESH)).sum())}/"
                 f"{int(((d['level'] == l) & (d[col] < THRESH)).sum())}" for l in rpb.LEVELS]
            print(f"{method:12s} {col:16s} silent/miscovered L3/L10/L20: {' '.join(s)}")
    print("\nplatt: mean interval width, mean |consensus - benchmark| and coverage by track and level:")
    p = df[df["method"] == "platt"]
    print(p.groupby(["track", "level"])[["width_pct", "mean_abs_div", "coverage_pct"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
