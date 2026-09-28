"""EDS-2026-0083 revision: null-refit coverage for Protocol A (worked case, P. leniusculus).

Protocol B's matched-perturbation null refits cover the clean benchmark at 0.98-1.00
(null_coverage.csv), so its coverage deficits are attributable to contamination.
Protocol A has a null only for the worked case (P. leniusculus, non-native range,
Random Forest, combined track; 30 null replicates per level in
data/rev_eds/protocol_a_null). Coverage of the companion study's deterministic
benchmark by the 2.5-97.5 % replicate interval, for the submitted, the locally
regenerated contaminated and the null ensembles, at L3/L10/L20.

Pre-registered reading (fixed before the run):
  null coverage >= 0.90 at every level: Protocol A's coverage deficits in this
      configuration are attributable to contamination, as for Protocol B;
  null coverage < 0.90 at any level: the 30-replicate interval misses the
      deterministic benchmark without contamination, so Protocol A coverage is read
      against its null rather than against 0.95, and the Protocol A silent counts
      carry that caveat.
Output: reports/rev_eds/worked_case_null_coverage.csv (ensemble labels: submitted,
local, null_refit; "null" itself is a pandas NA token).
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
LOCAL_ROOT = REPO / "data/rev_eds/protocol_a_local"
NULL_ROOT = REPO / "data/rev_eds/protocol_a_null"
OUT = REPO / "reports/rev_eds/worked_case_null_coverage.csv"
LEVELS = (3, 10, 20)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    warnings.simplefilter("ignore")
    ci = load_module("run_overpred_ci", REPO / "run_overpred_ci.py")
    from trustworthy_sdm.analysis import benchmark_for
    from trustworthy_sdm.conformal import CellID, load_ensemble

    bench = benchmark_for(ci.ENTITY, ci.ALGO, ci.TRACK, ci.PATHS)
    rows = []
    for level in LEVELS:
        cell = CellID(ci.ENTITY, ci.ALGO, ci.TRACK, axis="lowacc", level=level)
        for name, root in (("submitted", ci.SURFACES_ROOT), ("local", LOCAL_ROOT), ("null_refit", NULL_ROOT)):
            ens = load_ensemble(cell, root)
            idx = ens.index.intersection(bench.index)
            m = ens.loc[idx].to_numpy()
            b = bench.loc[idx].to_numpy()
            lo, hi = np.percentile(m, [2.5, 97.5], axis=1)
            rows.append({"level": level, "ensemble": name, "n_reps": m.shape[1], "n_subc": len(idx),
                         "coverage": float(np.mean((b >= lo) & (b <= hi))),
                         "mean_width": float(np.mean(hi - lo))})
    res = pd.DataFrame(rows)
    res.round(5).to_csv(OUT, index=False)
    print(f"{ci.ENTITY}, {ci.ALGO}, {ci.TRACK}: coverage of the deterministic benchmark (mean interval width)")
    for level in LEVELS:
        r = res[res["level"] == level].set_index("ensemble")
        print(f"  L{level:>2}: " + "  ".join(f"{e} {r.loc[e, 'coverage']:.3f} ({r.loc[e, 'mean_width']:.3f})"
                                         for e in ("submitted", "local", "null_refit")))
    j = pd.read_csv(REPO / "reports/rev_eds/item1_protocol_a_join.csv")
    txt = j.select_dtypes("object").astype(str).agg(" ".join, axis=1).str.lower()
    m = txt.str.contains("leniusculus") & txt.str.contains("combined") & txt.str.contains("random")
    cols = [c for c in ("level", "coverage_uncorrected") if c in j.columns]
    print("panel values for the same configuration (item1_protocol_a_join.csv):")
    print(j.loc[m, cols].to_string(index=False))
    nul = res[res["ensemble"] == "null_refit"]["coverage"]
    verdict = ("contamination-attributable, as for Protocol B" if (nul >= 0.90).all()
               else "Protocol A coverage to be read against its null; silent counts carry the caveat")
    print(f"pre-registered reading: null coverage {nul.min():.3f}-{nul.max():.3f} -> {verdict}")


if __name__ == "__main__":
    main()
