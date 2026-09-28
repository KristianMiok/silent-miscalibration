"""EDS-2026-0083 revision: null-refit coverage (Figure 1 reference) and worked-case flips at three thresholds.

1. Coverage of the matched-perturbation null refits (Protocol B): the envelope
   of the four null members (2.5-97.5 percentiles, as coverage_row) against the
   clean benchmark, uncalibrated and fixed-map Platt, per condition, next to the
   contaminated coverage -> null_coverage.csv. It is the coverage a perturbation
   of the same size yields without contamination: the empirical baseline for
   Figure 1 in place of a fixed tolerance.
2. Worked case (P. leniusculus, Protocol A): false-suitable and false-unsuitable
   counts at tau = 0.4, 0.5, 0.6 for the regenerated contaminated and the null
   ensembles (run_overprediction.py's own definition), and the net excess over
   the null -> worked_case_flips.csv.
"""
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
MASTER = Path(os.environ.get("TS_MASTER_CSV") or
              REPO.parent / "sdm-robustness/data/raw/combined_data_true_master.csv")
os.environ.setdefault("TS_MASTER_CSV", str(MASTER))
SURF = REPO / "data/replicate_surfaces_protocol_b"
NULL = REPO / "data/rev_eds/protocol_b_null"
PLATT = REPO / "data/rev_eds/protocol_b_platt"
NULL_PLATT = REPO / "data/rev_eds/protocol_b_null_platt"
LOCAL_A = REPO / "data/rev_eds/protocol_a_local"
NULL_A = REPO / "data/rev_eds/protocol_a_null"
OUT = REPO / "reports/rev_eds"
REPS = ["rep_00", "rep_01", "rep_02", "rep_03"]
DOMAIN_N = 37449


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def members(root, cell):
    return pd.concat([pd.read_parquet(root / cell / f"{r}.parquet").set_index("subc_id")["predicted_probability"]
                      for r in REPS], axis=1)


def main():
    warnings.simplefilter("ignore")
    rpb = load_module("run_protocol_b", REPO / "scripts/run_protocol_b.py")
    from trustworthy_sdm.analysis import ENTITY_NAME_TO_DIR as E2D

    rows = []
    for method, broot, nroot in (("uncalibrated", SURF, NULL), ("platt", PLATT, NULL_PLATT)):
        for e in rpb.ENTITIES:
            edir = E2D[e]
            for t in rpb.TRACKS:
                B = members(broot, f"{edir}__consensus__{t}__benchmark__L0").mean(axis=1)
                for lvl in rpb.LEVELS:
                    c = f"{edir}__consensus__{t}__lowacc__L{lvl}"
                    for source, root in (("contaminated", broot), ("null", nroot)):
                        M = members(root, c)
                        idx = M.index.intersection(B.index)
                        lo, hi = np.percentile(M.loc[idx].to_numpy(), [2.5, 97.5], axis=1)
                        b = B.loc[idx].to_numpy()
                        rows.append({"method": method, "source": source, "entity_dir": edir, "track": t,
                                     "level": lvl, "coverage": float(np.mean((b >= lo) & (b <= hi)))})
    cov = pd.DataFrame(rows)
    cov.round(5).to_csv(OUT / "null_coverage.csv", index=False)
    pan = pd.read_csv(OUT / "protocol_b_platt_panel.csv")
    mine = cov[cov["source"] == "contaminated"].merge(
        pan[["method", "entity_dir", "track", "level", "coverage"]], on=["method", "entity_dir", "track", "level"],
        suffixes=("", "_panel"))
    print(f"consistency with the calibrated panel: max |diff| {(mine['coverage'] - mine['coverage_panel']).abs().max():.2e}")
    print("\n1. coverage of contaminated ensembles vs null refits, mean over units:")
    print(cov.pivot_table(index=["method", "track", "level"], columns="source", values="coverage",
                          aggfunc="mean").round(3).to_string())
    for method in ("uncalibrated", "platt"):
        n = cov[(cov["method"] == method) & (cov["source"] == "null")]
        print(f"{method} null refits: minimum {n['coverage'].min():.3f}; below 0.90 in {int((n['coverage'] < 0.90).sum())} "
              f"of 72; below 0.95 in {int((n['coverage'] < 0.95).sum())} of 72")

    ro = load_module("run_overprediction", REPO / "run_overprediction.py")
    orig = ro.SURFACES_ROOT
    frows = []
    for tau in (0.4, 0.5, 0.6):
        for level in (3, 10, 20):
            rec = {"tau": tau, "level": level}
            for name, root in (("submitted", orig), ("local", LOCAL_A), ("null", NULL_A)):
                ro.SURFACES_ROOT = root
                f = ro.analyse_level(level, tau)[3]
                rec.update({f"{name}_false_suitable": f["n_false_suitable"],
                            f"{name}_false_unsuitable": f["n_false_unsuitable"],
                            f"{name}_net": f["net_overpred_cells"]})
            ro.SURFACES_ROOT = orig
            rec["excess_net"] = rec["local_net"] - rec["null_net"]
            rec["excess_net_pct"] = 100.0 * rec["excess_net"] / DOMAIN_N
            frows.append(rec)
    flips = pd.DataFrame(frows)
    flips.round(3).to_csv(OUT / "worked_case_flips.csv", index=False)
    print("\n2. worked case, net false-suitable sub-catchments (contaminated, null, excess):")
    print(flips[["tau", "level", "submitted_net", "local_net", "null_net", "excess_net", "excess_net_pct"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
